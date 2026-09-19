;;; nelisp-gpu-run.el --- dispatch GPU kernels from elisp via vkrun  -*- lexical-binding: t; -*-

;; Drives the generic Vulkan host (host/vkrun) from elisp: writes input
;; float buffers, runs a compiled kernel on the GPU, and reads the output
;; buffers back as float vectors.  Verification compares these against the
;; photon-tensor CPU oracle, so no per-kernel C host is needed.
;;
;; (This is still a subprocess host in C; Phase 3 replaces it with an
;; elisp-authored host.  vkrun is generic, so it is the last C host.)

;;; Code:

(require 'nelisp-gpu-compile)
(require 'nelisp-gpu-kernels)

(defvar nelisp-gpu-vkrun "host/vkrun"
  "Path to the generic Vulkan dispatcher binary (relative to `default-directory').")

(defvar nelisp-gpu--server-proc nil
  "Set by nelisp-gpu-server when the persistent GPU server is running.")
(declare-function nelisp-gpu-server-run "nelisp-gpu-server" (name buffers push groups))

(defun nelisp-gpu--bits-f32 (bits)
  "Decode a 32-bit IEEE-754 BITS pattern to an elisp float."
  (let* ((sign (if (zerop (logand bits #x80000000)) 1.0 -1.0))
         (exp (logand (ash bits -23) #xff))
         (mant (logand bits #x7fffff)))
    (cond ((and (= exp 0) (= mant 0)) (* sign 0.0))
          ((= exp 0)   (* sign (ldexp (/ mant 8388608.0) -126)))
          ((= exp 255) (* sign 1.0e30))
          (t (* sign (ldexp (+ 1.0 (/ mant 8388608.0)) (- exp 127)))))))

(defun nelisp-gpu--write-floats (vectors path)
  "Write each float vector in VECTORS (concatenated) to PATH as LE float32."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (dolist (v vectors)
      (dotimes (i (length v))
        (let ((b (nelisp-gpu--f32-bits (aref v i))))
          (insert (logand b #xff) (logand (ash b -8) #xff)
                  (logand (ash b -16) #xff) (logand (ash b -24) #xff)))))
    (let ((coding-system-for-write 'binary))
      (write-region (point-min) (point-max) path nil 'silent))))

(defun nelisp-gpu--read-floats (path total)
  "Read TOTAL LE float32 values from PATH into a vector."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (let ((coding-system-for-read 'binary))
      (insert-file-contents-literally path))
    (let ((v (make-vector total 0.0)) (p (point-min)))
      (dotimes (i total)
        (aset v i (nelisp-gpu--bits-f32
                   (logior (char-after p)
                           (ash (char-after (+ p 1)) 8)
                           (ash (char-after (+ p 2)) 16)
                           (ash (char-after (+ p 3)) 24))))
        (setq p (+ p 4)))
      v)))

(defun nelisp-gpu-dispatch (spv buffers push groups &optional local)
  "Run SPV on the GPU.  BUFFERS: list of float vectors (binding order;
output buffers passed as zero vectors).  PUSH: list of uints or nil.
GROUPS: workgroup count.  Returns the list of output float vectors."
  (let* ((sizes (mapcar #'length buffers))
         (total (apply #'+ sizes))
         (in (make-temp-file "nlgpu-in"))
         (out (make-temp-file "nlgpu-out"))
         (pushcsv (if push (mapconcat #'number-to-string push ",") "-"))
         (sizecsv (mapconcat #'number-to-string sizes ",")))
    (nelisp-gpu--write-floats buffers in)
    (let ((rc (call-process (expand-file-name nelisp-gpu-vkrun) nil nil nil
                            spv in out (number-to-string (or local 64))
                            (number-to-string groups) pushcsv sizecsv)))
      (unless (eq rc 0) (error "nelisp-gpu: vkrun failed rc=%S" rc)))
    (let ((all (nelisp-gpu--read-floats out total)) (res nil) (off 0))
      (dolist (s sizes (nreverse res))
        (push (substring all off (+ off s)) res)
        (setq off (+ off s))))))

(defun nelisp-gpu-run (name buffers push groups &optional local)
  "Run kernel NAME on the GPU and return its output float vectors.
Routes through the persistent server when it is up
\(`nelisp-gpu-server-start'); otherwise compiles NAME to SPIR-V and uses
the per-call `vkrun' host."
  (if (and nelisp-gpu--server-proc (process-live-p nelisp-gpu--server-proc))
      (nelisp-gpu-server-run name buffers push groups)
    (let ((spv (make-temp-file "nlgpu" nil ".spv")))
      (nelisp-gpu-write-kernel name spv)
      (unwind-protect
          (nelisp-gpu-dispatch spv buffers push groups local)
        (ignore-errors (delete-file spv))))))

(defun nelisp-gpu-gen (n fn)
  "Return a float vector of length N where element i is (funcall FN i)."
  (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (float (funcall fn i))) (setq i (1+ i)))
    v))

(provide 'nelisp-gpu-run)
;;; nelisp-gpu-run.el ends here
