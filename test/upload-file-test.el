;;; upload-file-test.el --- the server reads weights from disk  -*- lexical-binding: t -*-

;; `nelisp-gpu-server-upload-file' exists because the pipe, not the GPU, bounds
;; how fast a model reaches the device.  These checks pin the thing that would
;; make it useless: that the region it uploads is exactly the region asked for.
;; Every positive check is paired with a control that must disagree, so a stub
;; returning a zero buffer -- or one ignoring the offset -- fails here.

(require 'nelisp-gpu-run)
(require 'nelisp-gpu-server)
(require 'nelisp-gpu-kernels)

(defvar uft-pass 0)
(defvar uft-fail 0)

(defun uft-check (name ok fmt &rest args)
  (if ok (setq uft-pass (1+ uft-pass))
    (setq uft-fail (1+ uft-fail))
    (message "FAIL %s: %s" name (apply #'format fmt args))))

(defun uft-readback (handle n)
  "Copy the N floats of resident HANDLE back to the host through a copy kernel."
  (car (nelisp-gpu-server-batch
        (list (list 'res handle n)
              (cons 'in (make-vector n 0.0))
              (cons 'out n))
        (list (list 'add2 '(0 1 2) (list n) (/ (+ n 63) 64))))))

(defun uft-run ()
  (let* ((n 4096)
         (pad 12)                       ; a deliberately unaligned-looking prefix
         (vec (make-vector n 0.0))
         (path (make-temp-file "uft" nil ".bin")))
    (dotimes (i n)
      ;; Values a stub cannot guess and that survive float32 exactly.
      (aset vec i (float (- (* 3 i) 5000))))
    (let ((bytes (concat (make-string (* pad 4) ?\x00)
                         (nelisp-gpu--floats-bytes (list vec)))))
      (let ((coding-system-for-write 'binary))
        (write-region bytes nil path nil 'silent)))
    (nelisp-gpu-server-start)
    (unwind-protect
        (let* ((h (nelisp-gpu-server-upload-file path (* pad 4) (* n 4)))
               (got (uft-readback h n))
               (exact t) (worst 0.0))
          (dotimes (i n)
            (unless (= (aref got i) (aref vec i))
              (setq exact nil)
              (setq worst (max worst (abs (- (aref got i) (aref vec i)))))))
          (uft-check "region is exact" exact
                     "worst difference %s" worst)
          (uft-check "not all zero" (cl-some (lambda (x) (/= x 0.0)) (append got nil))
                     "every float read back as zero")

          ;; Control: the same call one float later must NOT match.  Without
          ;; this a server that ignored the offset would pass the check above
          ;; whenever pad happened to be 0.
          (let* ((h2 (nelisp-gpu-server-upload-file path (* (1+ pad) 4) (* (1- n) 4)))
                 (g2 (uft-readback h2 (1- n)))
                 (same t))
            (dotimes (i (1- n))
              (unless (= (aref g2 i) (aref vec i)) (setq same nil)))
            (uft-check "offset is honoured" (not same)
                       "shifting by one float changed nothing")
            (let ((shifted t))
              (dotimes (i (1- n))
                (unless (= (aref g2 i) (aref vec (1+ i))) (setq shifted nil)))
              (uft-check "shifted region is the shifted data" shifted
                         "offset+1 did not read the data at offset+1"))
            (nelisp-gpu-server-free h2))

          ;; The two upload paths must agree byte for byte.
          (let* ((raw (with-temp-buffer
                        (set-buffer-multibyte nil)
                        (let ((coding-system-for-read 'binary))
                          (insert-file-contents-literally path))
                        (buffer-substring-no-properties (1+ (* pad 4)) (point-max))))
                 (h3 (nelisp-gpu-server-upload-bytes raw))
                 (g3 (uft-readback h3 n))
                 (agree t))
            (dotimes (i n)
              (unless (= (aref g3 i) (aref got i)) (setq agree nil)))
            (uft-check "agrees with upload-bytes" agree
                       "the pipe path and the file path disagree")
            (nelisp-gpu-server-free h3))

          ;; A tail region, to catch an implementation that always reads from 0.
          (let* ((half (/ n 2))
                 (h4 (nelisp-gpu-server-upload-file
                      path (* (+ pad half) 4) (* half 4)))
                 (g4 (uft-readback h4 half))
                 (ok t))
            (dotimes (i half)
              (unless (= (aref g4 i) (aref vec (+ half i))) (setq ok nil)))
            (uft-check "tail region" ok "the second half did not read back")
            (nelisp-gpu-server-free h4))

          (nelisp-gpu-server-free h))
      (nelisp-gpu-server-stop)
      (delete-file path))
    (message "upload-file: %d passed, %d failed" uft-pass uft-fail)
    (when (> uft-fail 0) (kill-emacs 1))))

(require 'cl-lib)
(uft-run)
