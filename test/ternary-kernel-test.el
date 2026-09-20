;;; ternary-kernel-test.el --- the two-bit weight kernels -*- lexical-binding: t -*-

;; `ternary-rows' and its transpose read sixteen two-bit fields per word and a
;; scale per 128 columns.  Both are checked against the same arithmetic written
;; out in Elisp, and against each other through <W.x, g> = <x, W'.g>, which no
;; index swap survives.  Every positive check is paired with a control that
;; must disagree: a kernel returning zeros, or one ignoring the block scales,
;; fails here.

(require 'nelisp-gpu-run)
(require 'nelisp-gpu-server)
(require 'nelisp-gpu-kernels)
(require 'cl-lib)

(defvar tk-pass 0)
(defvar tk-fail 0)
(defvar tk-seed 20260921)

(defun tk-check (name ok fmt &rest args)
  (if ok (setq tk-pass (1+ tk-pass)) (setq tk-fail (1+ tk-fail)))
  (message "%-48s %s  %s" name (if ok "PASS" "FAIL") (apply #'format fmt args)))

(defun tk-err (got want)
  "Worst difference against the reference\='s own scale.

Per-element relative error is the wrong statistic for a vector that crosses
zero: an element near zero divides float32 noise by nothing and reports a
failure that is not one.  This divides by the reference\='s largest magnitude,
which is what \"the answer is right\" actually means here."
  (let ((scale 0.0) (worst 0.0))
    (dotimes (i (length want)) (setq scale (max scale (abs (aref want i)))))
    (dotimes (i (length want))
      (setq worst (max worst (abs (- (aref got i) (aref want i))))))
    (/ worst (max 1.0e-12 scale))))

(defun tk-rnd ()
  (setq tk-seed (mod (+ (* tk-seed 1103515245) 12345) 2147483648))
  (/ (float tk-seed) 2147483648.0))

(defun tk-pack (trits out cols wng)
  "Pack TRITS (OUT x COLS) into two-bit fields, sixteen per little-endian word."
  (let ((bytes (make-string (* out wng 4) 0)))
    (dotimes (o out)
      (dotimes (i cols)
        (let* ((v (aref trits (+ (* o cols) i)))
               (code (cond ((= v 1) 1) ((= v -1) 3) (t 0)))
               (idx (+ (* o wng 4) (ash i -2))))
          (aset bytes idx (logior (aref bytes idx)
                                  (ash code (* 2 (logand i 3))))))))
    bytes))

(defun tk-run ()
  (let* ((seq 3) (out 6) (cols 256) (bsize 128)
         (nblk (/ cols bsize)) (wng (/ cols 16))
         (trits (make-vector (* out cols) 0))
         (beta (make-vector (* out nblk) 0.0))
         (bias (make-vector out 0.0))
         (x (make-vector (* seq cols) 0.0))
         (g (make-vector (* seq out) 0.0)))
    (dotimes (o out)
      (aset bias o (- (* 0.4 (tk-rnd)) 0.2))
      (dotimes (b nblk) (aset beta (+ (* o nblk) b) (+ 0.01 (* 0.05 (tk-rnd)))))
      (dotimes (i cols)
        (aset trits (+ (* o cols) i) (- (truncate (* 3.0 (tk-rnd))) 1))))
    (dotimes (i (* seq cols)) (aset x i (- (* 2.0 (tk-rnd)) 1.0)))
    (dotimes (i (* seq out)) (aset g i (- (* 2.0 (tk-rnd)) 1.0)))
    (let ((wp (tk-pack trits out cols wng)))
      (nelisp-gpu-server-start)
      (unwind-protect
          (let ((h (nelisp-gpu-server-upload-bytes wp)))
            (tk-forward h seq out cols nblk wng trits beta bias x)
            (tk-transpose h seq out cols nblk wng trits beta x g)
            (tk-controls h seq out cols nblk wng beta bias x)
            (nelisp-gpu-server-free h))
        (nelisp-gpu-server-stop)))
    (message "ternary-kernel: %d passed, %d failed" tk-pass tk-fail)
    (when (> tk-fail 0) (kill-emacs 1))))

(defun tk-cpu-forward (trits beta bias x seq out cols nblk bsize)
  (let ((y (make-vector (* seq out) 0.0)))
    (dotimes (s seq)
      (dotimes (o out)
        (let ((acc 0.0))
          (dotimes (i cols)
            (setq acc (+ acc (* (aref trits (+ (* o cols) i))
                                (aref beta (+ (* o nblk) (/ i bsize)))
                                (aref x (+ (* s cols) i))))))
          (aset y (+ (* s out) o) (+ (aref bias o) acc)))))
    y))

(defun tk-forward (h seq out cols nblk wng trits beta bias x)
  (let* ((y (car (nelisp-gpu-server-batch
                  (list (cons 'in x)
                        (list 'res h (* out wng))
                        (cons 'in bias)
                        (cons 'in beta)
                        (cons 'out (* seq out)))
                  (list (list 'ternary-rows '(0 1 2 3 4)
                              (list seq out cols nblk wng)
                              (/ (+ (* seq out) 63) 64))))))
         (want (tk-cpu-forward trits beta bias x seq out cols nblk 128))
         (worst (tk-err y want)))
    (tk-check "ternary-rows matches the same sum in Elisp"
              (< worst 1.0e-5) "worst %.3e of the reference's scale" worst)
    (let ((nonzero (cl-some (lambda (v) (/= v 0.0)) (append y nil))))
      (tk-check "control: the output is not all zero" nonzero
                (if nonzero "it is not" "every output came back zero")))))

(defun tk-transpose (h seq out cols nblk wng trits beta x g)
  (let* ((xt (car (nelisp-gpu-server-batch
                   (list (list 'res h (* out wng))
                         (cons 'in beta)
                         (cons 'in g)
                         (cons 'out (* seq cols)))
                   (list (list 'ternary-rows-t '(0 1 2 3)
                               (list out cols nblk wng seq)
                               (/ (+ (* seq cols) 63) 64))))))
         (want (make-vector (* seq cols) 0.0)))
    (dotimes (s seq)
      (dotimes (i cols)
        (let ((acc 0.0))
          (dotimes (o out)
            (setq acc (+ acc (* (aref trits (+ (* o cols) i))
                                (aref beta (+ (* o nblk) (/ i 128)))
                                (aref g (+ (* s out) o))))))
          (aset want (+ (* s cols) i) acc))))
    (tk-check "ternary-rows-t matches the same sum in Elisp"
              (< (tk-err xt want) 1.0e-5)
              "worst %.3e of the reference's scale" (tk-err xt want))
    ;; and the two kernels against each other, which needs no reference
    (let* ((y (car (nelisp-gpu-server-batch
                    (list (cons 'in x)
                          (list 'res h (* out wng))
                          (cons 'in (make-vector out 0.0))
                          (cons 'in beta)
                          (cons 'out (* seq out)))
                    (list (list 'ternary-rows '(0 1 2 3 4)
                                (list seq out cols nblk wng)
                                (/ (+ (* seq out) 63) 64))))))
           (l 0.0) (r 0.0))
      (dotimes (i (* seq out)) (setq l (+ l (* (aref y i) (aref g i)))))
      (dotimes (i (* seq cols)) (setq r (+ r (* (aref x i) (aref xt i)))))
      (tk-check "the two kernels agree: <W.x, g> = <x, W'.g>"
                (< (/ (abs (- l r)) (max 1.0e-6 (abs l))) 1.0e-5)
                "%.6f against %.6f" l r))))

(defun tk-controls (h seq out cols nblk wng beta bias x)
  "The block scales must be load-bearing, or the checks above pass a constant."
  (let* ((b2 (copy-sequence beta))
         (run (lambda (bb)
                (car (nelisp-gpu-server-batch
                      (list (cons 'in x)
                            (list 'res h (* out wng))
                            (cons 'in bias)
                            (cons 'in bb)
                            (cons 'out (* seq out)))
                      (list (list 'ternary-rows '(0 1 2 3 4)
                                  (list seq out cols nblk wng)
                                  (/ (+ (* seq out) 63) 64))))))))
    (aset b2 1 (* 3.0 (aref b2 1)))     ; row 0, second block only
    (let* ((y0 (funcall run beta)) (y1 (funcall run b2))
           (moved 0) (spilled 0))
      (dotimes (s seq)
        (when (/= (aref y0 (* s out)) (aref y1 (* s out))) (setq moved (1+ moved)))
        (dotimes (o (1- out))
          (unless (= (aref y0 (+ (* s out) o 1)) (aref y1 (+ (* s out) o 1)))
            (setq spilled (1+ spilled)))))
      (tk-check "control: changing one block's scale moves that row"
                (= moved seq) "%d of %d rows moved" moved seq)
      (tk-check "control: and moves no other row"
                (= spilled 0) "%d other outputs changed" spilled))))

(tk-run)
