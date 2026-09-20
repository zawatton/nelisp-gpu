;;; nelisp-gpu-server.el --- persistent GPU server client  -*- lexical-binding: t; -*-

;; Talks to host/vkserver, a long-lived Vulkan process that keeps the
;; device + per-kernel pipelines alive, so dispatches avoid per-op
;; subprocess spawn, Vulkan re-init and temp-file IO.  Binary
;; request/response over the process stdin/stdout (stderr is kept on a
;; separate pipe so it never corrupts the protocol).  When the server is
;; running, `nelisp-gpu-run' routes through it automatically.
;;
;; Resident buffers: weights can be uploaded once with
;; `nelisp-gpu-server-resident' and then referenced by handle in
;; `nelisp-gpu-server-run2', so a model's weights are encoded + uploaded
;; a single time instead of per op per token.

;;; Code:

(require 'nelisp-gpu-run)   ; f32 encode/decode, write-kernel

(defconst nelisp-gpu--op-run 0)
(defconst nelisp-gpu--op-upload 1)
(defconst nelisp-gpu--op-free 2)
(defconst nelisp-gpu--op-batch 3)
(defconst nelisp-gpu--op-compile 4)
(defconst nelisp-gpu--op-run-compiled 5)
(defconst nelisp-gpu--op-free-compiled 6)
(defconst nelisp-gpu--op-write-resident 7)
(defconst nelisp-gpu--op-upload-file 8)

(defvar nelisp-gpu-server-bin "host/vkserver"
  "Path to the persistent server binary (relative to `default-directory').")
(defvar nelisp-gpu--server-proc nil)
(defvar nelisp-gpu--server-out "")
(defvar nelisp-gpu--server-kdir nil)
(defvar nelisp-gpu--resident nil
  "eq hash-table mapping a float vector to its resident GPU handle.")

(defun nelisp-gpu--u32-bytes (n)
  (unibyte-string (logand n #xff) (logand (ash n -8) #xff)
                  (logand (ash n -16) #xff) (logand (ash n -24) #xff)))

(defun nelisp-gpu--u32-at (buf p)
  (logior (aref buf p) (ash (aref buf (+ p 1)) 8)
          (ash (aref buf (+ p 2)) 16) (ash (aref buf (+ p 3)) 24)))

(defun nelisp-gpu--floats-bytes (vectors)
  "Concatenate float VECTORS as little-endian float32 unibyte bytes."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (dolist (v vectors)
      (dotimes (i (length v))
        (let ((b (nelisp-gpu--f32-bits (aref v i))))
          (insert (logand b #xff) (logand (ash b -8) #xff)
                  (logand (ash b -16) #xff) (logand (ash b -24) #xff)))))
    (buffer-string)))

(defun nelisp-gpu-server-up-p ()
  "Return non-nil when the persistent GPU server is running."
  (and nelisp-gpu--server-proc (process-live-p nelisp-gpu--server-proc)))

(defun nelisp-gpu-server-start (&optional kernels)
  "Compile KERNELS to a temp dir and start the persistent GPU server."
  (nelisp-gpu-server-stop)
  (let ((dir (make-temp-file "nlgpu-k" t)))
    (dolist (k (or kernels '(matmul linear softmax gelu layernorm vadd
                             transpose scale causal-mask slice-cols set-cols
                             sub-scale colsum gelu-bwd sgd
                             rmsnorm-istd rmsnorm-fwd rmsnorm-dx rmsnorm-dgamma
                             rope-apply silu silu-bwd mul softmax-bwd ce-grad
                             topk-mask scale-rows rowdot
                             embed-gather embed-bwd ce-grad-idx adam
                             attn-scores attn-context attn-sc-dq attn-sc-dk
                             attn-ctx-dp attn-ctx-dv
                             matmul-at silu-mul silu-mul-bwd
                             sumsq-acc clip-scale
                             decode-rope cache-append decode-attn
                             decode-rope-b cache-append-b decode-attn-b
                             cache-append-ring decode-attn-stream
                             decode-rope-b-v cache-append-paged-v decode-attn-paged-v
                             block-copy tree-attn qkv-append attn-stream-entries
                             absmean-acc quant-w quant-act bitlinear-packed bitlinear-packed-v
                             dp4a-dot bitlinear-dp4a bitlinear-dp4a-1f
                             bitlinear-dp4a-rows dp4a-rows-t
                             attn-causal-gqa pack-act-rows
                             rmsnorm-heads rope-half add2
                             gather-spike scatter-spike
                             cache-append-paged decode-attn-paged)))
      (nelisp-gpu-write-kernel k (expand-file-name (format "%s.spv" k) dir)))
    (setq nelisp-gpu--server-kdir dir
          nelisp-gpu--server-out ""
          nelisp-gpu--resident (make-hash-table :test 'eq))
    (setq nelisp-gpu--server-proc
          (make-process
           :name "vkserver"
           :command (list (expand-file-name nelisp-gpu-server-bin) dir)
           :connection-type 'pipe
           :coding 'binary
           :noquery t
           :stderr (get-buffer-create " *vkserver-stderr*")
           :filter (lambda (_p s)
                     (setq nelisp-gpu--server-out
                           (concat nelisp-gpu--server-out s)))))
    nelisp-gpu--server-proc))

(defun nelisp-gpu-server-stop ()
  "Stop the persistent GPU server, if running, and drop resident handles."
  (when (and nelisp-gpu--server-proc (process-live-p nelisp-gpu--server-proc))
    (ignore-errors (delete-process nelisp-gpu--server-proc)))
  (setq nelisp-gpu--server-proc nil
        nelisp-gpu--resident nil))

(defun nelisp-gpu--server-xfer (req expected)
  "Send REQ bytes; wait until EXPECTED response bytes arrive; return them."
  (setq nelisp-gpu--server-out "")
  (process-send-string nelisp-gpu--server-proc req)
  (while (< (length nelisp-gpu--server-out) expected)
    (unless (process-live-p nelisp-gpu--server-proc) (error "vkserver died"))
    (accept-process-output nelisp-gpu--server-proc 10))
  nelisp-gpu--server-out)

(defun nelisp-gpu-server-upload (vec)
  "Upload float VEC to a persistent GPU buffer; return its integer handle."
  (let* ((req (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-upload)
                      (nelisp-gpu--u32-bytes (length vec))
                      (nelisp-gpu--floats-bytes (list vec))))
         (buf (nelisp-gpu--server-xfer req 8)))
    (unless (zerop (nelisp-gpu--u32-at buf 0)) (error "vkserver upload error"))
    (nelisp-gpu--u32-at buf 4)))

(defun nelisp-gpu-server-upload-u32 (uvec)
  "Upload UVEC (uint32 integers) to a persistent GPU buffer with the words stored
verbatim (e.g. four int8 lanes packed per word, read back in a kernel via
`bitcast-u').  Returns the integer handle."
  (let* ((req (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-upload)
                      (nelisp-gpu--u32-bytes (length uvec))
                      (mapconcat #'nelisp-gpu--u32-bytes (append uvec nil) "")))
         (buf (nelisp-gpu--server-xfer req 8)))
    (unless (zerop (nelisp-gpu--u32-at buf 0)) (error "vkserver upload-u32 error"))
    (nelisp-gpu--u32-at buf 4)))

(defun nelisp-gpu-server-upload-bytes (bytes)
  "Upload BYTES to a persistent GPU buffer verbatim; return its integer handle.
BYTES is a unibyte string holding little-endian uint32 words -- e.g. four int8
lanes per word, read back in a kernel with `bitcast-u'.

This exists because `nelisp-gpu-server-upload-u32' takes a vector of Elisp
integers and lists it before encoding, which is fine for a kernel's worth of
words and hopeless for a model's: an imported Qwen3-0.6B is ~149M words, so
that path would build a 1.2 GB vector and a 2.4 GB list in order to send bytes
that are already laid out correctly on disk.  Here the caller hands over the
bytes it read and nothing is allocated per word."
  (unless (stringp bytes)
    (error "upload-bytes: BYTES must be a string, got %S" (type-of bytes)))
  (when (multibyte-string-p bytes)
    (error "upload-bytes: BYTES must be unibyte (got a multibyte string, \
which would be re-encoded rather than sent verbatim)"))
  (unless (zerop (% (length bytes) 4))
    (error "upload-bytes: %d bytes is not a whole number of uint32 words"
           (length bytes)))
  (let* ((req (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-upload)
                      (nelisp-gpu--u32-bytes (/ (length bytes) 4))
                      bytes))
         (buf (nelisp-gpu--server-xfer req 8)))
    (unless (zerop (nelisp-gpu--u32-at buf 0))
      (error "vkserver upload-bytes error"))
    (nelisp-gpu--u32-at buf 4)))

(defun nelisp-gpu-server-upload-file (path offset nbytes)
  "Upload the NBYTES at OFFSET of PATH into a resident buffer; return its handle.

The bytes are read by the server and never enter Emacs.  `process-send-string'
moves about 3 MB/s into a pipe -- measured on this machine against 147 MB/s for
the same string written to a file -- so a model's worth of weights sent through
`nelisp-gpu-server-upload-bytes' is bounded by the pipe and not by the GPU.
OFFSET and NBYTES are byte counts and may exceed 32 bits; NBYTES must be a whole
number of uint32 words."
  (unless (zerop (% nbytes 4))
    (error "upload-file: %d bytes is not a whole number of uint32 words" nbytes))
  (let* ((full (encode-coding-string (expand-file-name path) 'utf-8 t))
         (req (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-upload-file)
                      (nelisp-gpu--u32-bytes (length full))
                      full
                      (nelisp-gpu--u32-bytes (logand offset #xffffffff))
                      (nelisp-gpu--u32-bytes (ash offset -32))
                      (nelisp-gpu--u32-bytes (/ nbytes 4))))
         (buf (nelisp-gpu--server-xfer req 8)))
    (unless (zerop (nelisp-gpu--u32-at buf 0))
      (error "vkserver upload-file error (%s +%d, %d bytes)" path offset nbytes))
    (nelisp-gpu--u32-at buf 4)))

(defun nelisp-gpu-server-free (handle)
  "Free the resident GPU buffer HANDLE on the server."
  (let ((buf (nelisp-gpu--server-xfer
              (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-free)
                      (nelisp-gpu--u32-bytes handle))
              4)))
    (unless (zerop (nelisp-gpu--u32-at buf 0)) (error "vkserver free error"))))

(defun nelisp-gpu-server-write-resident (handle vec)
  "Overwrite resident buffer HANDLE's contents with float VEC, in place.
Lets a compiled batch be re-run on fresh input (e.g. the next training window's
one-hot tokens/targets) without recompiling."
  (let ((buf (nelisp-gpu--server-xfer
              (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-write-resident)
                      (nelisp-gpu--u32-bytes handle)
                      (nelisp-gpu--u32-bytes (length vec))
                      (nelisp-gpu--floats-bytes (list vec)))
              4)))
    (unless (zerop (nelisp-gpu--u32-at buf 0)) (error "vkserver write-resident error"))))

(defun nelisp-gpu-server-read-resident (handle n)
  "Read N floats back from resident buffer HANDLE to the host (via the identity
`scale'-by-1 kernel).  Used to checkpoint resident state (weights, optimiser
moments) without round-tripping through host copies."
  (car (nelisp-gpu-server-run2 'scale
                               (list (list 'res handle n) (cons 'in (vector 1.0)) (cons 'out n))
                               (list n) (/ (+ n 63) 64))))

(defun nelisp-gpu-server-resident (vec)
  "Return a cached resident handle for float VEC, uploading on first use.
The handle is keyed on VEC's object identity, so stable weight vectors
upload exactly once for the life of the server."
  (unless nelisp-gpu--resident
    (setq nelisp-gpu--resident (make-hash-table :test 'eq)))
  (or (gethash vec nelisp-gpu--resident)
      (puthash vec (nelisp-gpu-server-upload vec) nelisp-gpu--resident)))

(defun nelisp-gpu-server-invalidate (vec)
  "Drop VEC's cached resident handle, freeing it on the server, so that the
next `nelisp-gpu-server-resident' call re-uploads VEC's *current* contents.
The resident cache is keyed on VEC's object identity and cannot observe an
in-place mutation, so a mutated weight (e.g. after an SGD step) must be
invalidated explicitly or the GPU keeps using the stale upload."
  (when nelisp-gpu--resident
    (let ((h (gethash vec nelisp-gpu--resident)))
      (when h
        (remhash vec nelisp-gpu--resident)
        (ignore-errors (nelisp-gpu-server-free h))))))

(defun nelisp-gpu-server-run2 (name descs push groups)
  "Dispatch kernel NAME with per-buffer DESCS; return output vectors.
Each desc (binding order) is one of:
  (in . VEC)        inline input  (sent, not returned)
  (res HANDLE SIZE) resident      (referenced by HANDLE, not returned)
  (out . SIZE)      output        (zeroed, returned)
  (inout . VEC)     inline inout  (sent and returned)
Returns the vectors for the out/inout descs, in binding order."
  (let* ((nm (if (symbolp name) (symbol-name name) name))
         (parts nil) (inline nil) (outsizes nil))
    (push (nelisp-gpu--u32-bytes nelisp-gpu--op-run) parts)
    (push (nelisp-gpu--u32-bytes (length nm)) parts)
    (push (string-to-unibyte nm) parts)
    (push (nelisp-gpu--u32-bytes (length descs)) parts)
    (dolist (d descs)
      (pcase (car d)
        ('in    (let ((v (cdr d)))
                  (push (nelisp-gpu--u32-bytes (length v)) parts)
                  (push (nelisp-gpu--u32-bytes 0) parts)
                  (push v inline)))
        ('res   (let ((h (nth 1 d)) (sz (nth 2 d)))
                  (push (nelisp-gpu--u32-bytes sz) parts)
                  (push (nelisp-gpu--u32-bytes 1) parts)
                  (push (nelisp-gpu--u32-bytes h) parts)))
        ('out   (let ((sz (cdr d)))
                  (push (nelisp-gpu--u32-bytes sz) parts)
                  (push (nelisp-gpu--u32-bytes 2) parts)
                  (push sz outsizes)))
        ('inout (let ((v (cdr d)))
                  (push (nelisp-gpu--u32-bytes (length v)) parts)
                  (push (nelisp-gpu--u32-bytes 3) parts)
                  (push v inline)
                  (push (length v) outsizes)))))
    (setq parts (nreverse parts)
          inline (nreverse inline)
          outsizes (nreverse outsizes))
    (setq parts (append parts
                        (list (nelisp-gpu--u32-bytes (length push)))
                        (mapcar #'nelisp-gpu--u32-bytes push)
                        (list (nelisp-gpu--u32-bytes groups))
                        (list (nelisp-gpu--floats-bytes inline))))
    (let* ((out-total (apply #'+ outsizes))
           (buf (nelisp-gpu--server-xfer (apply #'concat parts) (+ 4 (* out-total 4)))))
      (unless (zerop (nelisp-gpu--u32-at buf 0)) (error "vkserver run error"))
      (let ((res nil) (off 0))
        (dolist (s outsizes (nreverse res))
          (let ((v (make-vector s 0.0)) (i 0))
            (while (< i s)
              (aset v i (nelisp-gpu--bits-f32 (nelisp-gpu--u32-at buf (+ 4 (* (+ off i) 4)))))
              (setq i (1+ i)))
            (push v res)
            (setq off (+ off s))))))))

(defun nelisp-gpu-server-run (name buffers push groups)
  "Dispatch kernel NAME on the persistent server; return all buffers back.
Generic path: every buffer is treated as inline inout, matching the
legacy contract used by `nelisp-gpu-run'."
  (nelisp-gpu-server-run2 name (mapcar (lambda (v) (cons 'inout v)) buffers)
                          push groups))

(defun nelisp-gpu--batch-body (slots disps)
  "Encode the batch wire body (everything after the opcode).
Return (BODY-STRING . OUTSIZES) where OUTSIZES is in slot order."
  (let ((parts nil) (inline nil) (outsizes nil))
    (push (nelisp-gpu--u32-bytes (length slots)) parts)
    (dolist (s slots)
      (pcase (car s)
        ('in  (push (nelisp-gpu--u32-bytes 0) parts)
              (push (nelisp-gpu--u32-bytes (length (cdr s))) parts)
              (push (cdr s) inline))
        ('res (push (nelisp-gpu--u32-bytes 1) parts)
              (push (nelisp-gpu--u32-bytes (nth 2 s)) parts)
              (push (nelisp-gpu--u32-bytes (nth 1 s)) parts))
        ('tmp (push (nelisp-gpu--u32-bytes 2) parts)
              (push (nelisp-gpu--u32-bytes (cdr s)) parts))
        ('out (push (nelisp-gpu--u32-bytes 3) parts)
              (push (nelisp-gpu--u32-bytes (cdr s)) parts)
              (push (cdr s) outsizes))))
    (push (nelisp-gpu--u32-bytes (length disps)) parts)
    (dolist (d disps)
      (let* ((nm0 (nth 0 d)) (nm (if (symbolp nm0) (symbol-name nm0) nm0))
             (slotidx (nth 1 d)) (pushv (nth 2 d)) (grp (nth 3 d)))
        (push (nelisp-gpu--u32-bytes (length nm)) parts)
        (push (string-to-unibyte nm) parts)
        (push (nelisp-gpu--u32-bytes (length slotidx)) parts)
        (dolist (si slotidx) (push (nelisp-gpu--u32-bytes si) parts))
        (push (nelisp-gpu--u32-bytes (length pushv)) parts)
        (dolist (pv pushv) (push (nelisp-gpu--u32-bytes pv) parts))
        (push (nelisp-gpu--u32-bytes grp) parts)))
    (setq parts (nreverse parts) inline (nreverse inline))
    (cons (apply #'concat (append parts (list (nelisp-gpu--floats-bytes inline))))
          (nreverse outsizes))))

(defun nelisp-gpu--decode-outs (buf outsizes)
  "Decode the OUTSIZES float vectors from response BUF (after the status u32)."
  (let ((res nil) (off 0))
    (dolist (s outsizes (nreverse res))
      (let ((v (make-vector s 0.0)) (i 0))
        (while (< i s)
          (aset v i (nelisp-gpu--bits-f32 (nelisp-gpu--u32-at buf (+ 4 (* (+ off i) 4)))))
          (setq i (1+ i)))
        (push v res) (setq off (+ off s))))))

(defun nelisp-gpu-server-batch (slots disps)
  "Run a fused batch of dispatches in one GPU command buffer; return outputs.
SLOTS (index order) each: (in . VEC) | (res HANDLE SIZE) | (tmp . SIZE)
| (out . SIZE).  Intermediate (tmp) tensors stay resident on the GPU and
are never marshalled to the host.  DISPS each: (NAME (SLOT-IDX...)
(PUSH...) GROUPS); a memory barrier is inserted between consecutive
dispatches.  Returns the vectors for the out slots, in slot order."
  (let* ((bo (nelisp-gpu--batch-body slots disps)) (outsizes (cdr bo))
         (out-total (apply #'+ outsizes))
         (buf (nelisp-gpu--server-xfer
               (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-batch) (car bo))
               (+ 4 (* out-total 4)))))
    (unless (zerop (nelisp-gpu--u32-at buf 0)) (error "vkserver batch error"))
    (nelisp-gpu--decode-outs buf outsizes)))

(defun nelisp-gpu-server-compile (slots disps)
  "Compile a fixed batch (SLOTS, DISPS) into a persistent, pre-recorded command
buffer on the server.  Return (HANDLE . OUTSIZES); run it with
`nelisp-gpu-server-run-compiled' and release with `-free-compiled'.  Use this
when the graph is constant across many runs (e.g. a training loop): only the
resident weight buffers change in place, so the command buffer is recorded once
and just re-submitted -- avoiding per-step protocol re-send, buffer allocation
and descriptor rebuild."
  (let* ((bo (nelisp-gpu--batch-body slots disps))
         (buf (nelisp-gpu--server-xfer
               (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-compile) (car bo)) 8)))
    (unless (zerop (nelisp-gpu--u32-at buf 0)) (error "vkserver compile error"))
    (cons (nelisp-gpu--u32-at buf 4) (cdr bo))))

(defun nelisp-gpu-server-run-compiled (handle outsizes)
  "Re-submit the compiled batch HANDLE; return its OUTSIZES output vectors.
Tmp/grad buffers are zeroed on-GPU by the recorded command buffer; resident
weights persist and update in place."
  (let* ((out-total (apply #'+ outsizes))
         (buf (nelisp-gpu--server-xfer
               (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-run-compiled)
                       (nelisp-gpu--u32-bytes handle))
               (+ 4 (* out-total 4)))))
    (unless (zerop (nelisp-gpu--u32-at buf 0)) (error "vkserver run-compiled error"))
    (nelisp-gpu--decode-outs buf outsizes)))

(defun nelisp-gpu-server-free-compiled (handle)
  "Free the compiled batch HANDLE on the server (buffers, descriptors, cmd)."
  (let ((buf (nelisp-gpu--server-xfer
              (concat (nelisp-gpu--u32-bytes nelisp-gpu--op-free-compiled)
                      (nelisp-gpu--u32-bytes handle)) 4)))
    (unless (zerop (nelisp-gpu--u32-at buf 0)) (error "vkserver free-compiled error"))))

(provide 'nelisp-gpu-server)
;;; nelisp-gpu-server.el ends here
