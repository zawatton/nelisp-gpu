;;; verify.el --- verify GPU kernels against the photon-tensor CPU oracle  -*- lexical-binding: t; -*-
;; Run from the project root:
;;   emacs -Q --batch -L lisp -l test/verify.el
;; Requires ../nelisp-photon/lisp/photon-tensor.el as the reference oracle
;; and host/vkrun built.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nelisp-gpu-run)
(require 'photon-tensor)

(defun verify--max-err (a b)
  (let ((m 0.0) (i 0) (n (length a)))
    (while (< i n)
      (let ((e (abs (- (aref a i) (aref b i))))) (when (> e m) (setq m e)))
      (setq i (1+ i)))
    m))

(defun verify--ck (name err tol)
  (princ (format "%-12s max_err=%.3e  %s\n" name err (if (< err tol) "PASS" "FAIL"))))

;; matmul  (32x24) * (24x16)
(let* ((M 32) (K 24) (N 16)
       (A (nelisp-gpu-gen (* M K) (lambda (i) (* 0.1 (- (mod i 7) 3)))))
       (B (nelisp-gpu-gen (* K N) (lambda (i) (* 0.1 (- (mod i 5) 2)))))
       (C0 (make-vector (* M N) 0.0))
       (ref (photon-tensor-data
             (photon-tensor-matmul (photon-tensor (list M K) A)
                                   (photon-tensor (list K N) B))))
       (got (nth 2 (nelisp-gpu-run 'matmul (list A B C0)
                                   (list M K N) (/ (+ (* M N) 63) 64)))))
  (verify--ck "matmul" (verify--max-err got ref) 1e-3))

;; softmax  (24x40) row-wise
(let* ((M 24) (N 40)
       (A (nelisp-gpu-gen (* M N) (lambda (i) (* 0.2 (- (mod i 11) 5)))))
       (C0 (make-vector (* M N) 0.0))
       (ref (photon-tensor-data
             (photon-tensor-softmax-rows (photon-tensor (list M N) A))))
       (got (nth 1 (nelisp-gpu-run 'softmax (list A C0)
                                   (list M N) (/ (+ M 63) 64)))))
  (verify--ck "softmax" (verify--max-err got ref) 1e-4))

;; gelu  n=100
(let* ((n 100)
       (A (nelisp-gpu-gen n (lambda (i) (* 0.1 (- i 50)))))
       (C0 (make-vector n 0.0))
       (ref (photon-tensor-data (photon-tensor-gelu (photon-tensor (list 1 n) A))))
       (got (nth 1 (nelisp-gpu-run 'gelu (list A C0) (list n) (/ (+ n 63) 64)))))
  (verify--ck "gelu" (verify--max-err got ref) 1e-4))

;; layernorm  (24x40) row-wise, gamma=1 beta=0
(let* ((M 24) (N 40)
       (A (nelisp-gpu-gen (* M N) (lambda (i) (* 0.3 (- (mod i 13) 6)))))
       (C0 (make-vector (* M N) 0.0))
       (g (make-vector N 1.0)) (be (make-vector N 0.0))
       (ref (photon-tensor-data
             (photon-tensor-layernorm-rows (photon-tensor (list M N) A)
                                           (photon-tensor (list N) g)
                                           (photon-tensor (list N) be))))
       (got (nth 1 (nelisp-gpu-run 'layernorm (list A C0)
                                   (list M N) (/ (+ M 63) 64)))))
  (verify--ck "layernorm" (verify--max-err got ref) 1e-3))

;; linear  X(20x12) * W(16x12)^T + B(16)
(let* ((M 20) (IN 12) (OUT 16)
       (X (nelisp-gpu-gen (* M IN) (lambda (i) (* 0.1 (- (mod i 7) 3)))))
       (W (nelisp-gpu-gen (* OUT IN) (lambda (i) (* 0.1 (- (mod i 5) 2)))))
       (B (nelisp-gpu-gen OUT (lambda (i) (* 0.05 (- (mod i 3) 1)))))
       (C0 (make-vector (* M OUT) 0.0))
       (ref (photon-tensor-data
             (photon-tensor-linear (photon-tensor (list M IN) X)
                                   (photon-tensor (list OUT IN) W)
                                   (photon-tensor (list OUT) B))))
       (got (nth 3 (nelisp-gpu-run 'linear (list X W B C0)
                                   (list M IN OUT) (/ (+ (* M OUT) 63) 64)))))
  (verify--ck "linear" (verify--max-err got ref) 1e-3))

;; matmul-tiled  (20x24)*(24x12) -- non-multiples of 16 to exercise tile guards
(let* ((M 20) (K 24) (N 12)
       (A (nelisp-gpu-gen (* M K) (lambda (i) (* 0.1 (- (mod i 7) 3)))))
       (B (nelisp-gpu-gen (* K N) (lambda (i) (* 0.1 (- (mod i 5) 2)))))
       (C0 (make-vector (* M N) 0.0))
       (ref (photon-tensor-data
             (photon-tensor-matmul (photon-tensor (list M K) A)
                                   (photon-tensor (list K N) B))))
       (got (nth 2 (nelisp-gpu-run 'matmul-tiled (list A B C0) (list M K N)
                                   (* (/ (+ M 15) 16) (/ (+ N 15) 16))))))
  (verify--ck "matmul-tiled" (verify--max-err got ref) 1e-3))

;; transpose  (5x7) -> (7x5)
(let* ((R 5) (Co 7)
       (A (nelisp-gpu-gen (* R Co) (lambda (i) (float i))))
       (C0 (make-vector (* R Co) 0.0))
       (got (nth 1 (nelisp-gpu-run 'transpose (list A C0) (list R Co)
                                   (/ (+ (* R Co) 63) 64))))
       (ref (make-vector (* R Co) 0.0)))
  (dotimes (i R) (dotimes (j Co) (aset ref (+ (* j R) i) (aref A (+ (* i Co) j)))))
  (verify--ck "transpose" (verify--max-err got ref) 1e-6))

;; scale  A*k via a 1-element scalar buffer
(let* ((n 50) (k 0.3)
       (A (nelisp-gpu-gen n (lambda (i) (* 0.1 (- i 25)))))
       (S (vector k)) (C0 (make-vector n 0.0))
       (got (nth 2 (nelisp-gpu-run 'scale (list A S C0) (list n) (/ (+ n 63) 64))))
       (ref (make-vector n 0.0)))
  (dotimes (i n) (aset ref i (* (aref A i) k)))
  (verify--ck "scale" (verify--max-err got ref) 1e-5))

;; causal-mask  upper triangle (j>i) -> -1e30, in place on (6x6)
(let* ((Nn 6)
       (S (nelisp-gpu-gen (* Nn Nn) (lambda (i) (float (1+ i)))))
       (got (nth 0 (nelisp-gpu-run 'causal-mask (list (copy-sequence S))
                                   (list Nn) (/ (+ (* Nn Nn) 63) 64))))
       (ref (copy-sequence S)))
  (dotimes (i Nn) (dotimes (j Nn) (when (> j i) (aset ref (+ (* i Nn) j) -1.0e30))))
  (verify--ck "causal-mask" (verify--max-err got ref) 1e25))

;; slice-cols  (4x8) cols [2,5) -> (4x3)
(let* ((seq 4) (dim 8) (hd 3) (c0 2)
       (A (nelisp-gpu-gen (* seq dim) (lambda (i) (float i))))
       (C0 (make-vector (* seq hd) 0.0))
       (got (nth 1 (nelisp-gpu-run 'slice-cols (list A C0) (list seq dim hd c0)
                                   (/ (+ (* seq hd) 63) 64))))
       (ref (make-vector (* seq hd) 0.0)))
  (dotimes (i seq) (dotimes (j hd) (aset ref (+ (* i hd) j) (aref A (+ (* i dim) c0 j)))))
  (verify--ck "slice-cols" (verify--max-err got ref) 1e-6))

;; set-cols  write (4x3) into cols [2,5) of (4x8), in place
(let* ((seq 4) (dim 8) (hd 3) (c0 2)
       (D (nelisp-gpu-gen (* seq dim) (lambda (i) (float (* 10 i)))))
       (Sx (nelisp-gpu-gen (* seq hd) (lambda (i) (float (- i)))))
       (got (nth 0 (nelisp-gpu-run 'set-cols (list (copy-sequence D) Sx)
                                   (list seq dim hd c0) (/ (+ (* seq hd) 63) 64))))
       (ref (copy-sequence D)))
  (dotimes (i seq) (dotimes (j hd) (aset ref (+ (* i dim) c0 j) (aref Sx (+ (* i hd) j)))))
  (verify--ck "set-cols" (verify--max-err got ref) 1e-6))

;; matmul-reg  (20x24)*(24x12) -- register-blocked 4x4 micro-tile.
;; Fast path for aligned shapes: M and N must be multiples of 4 (1024 et al).
;; Group count = ceil(ceil(M/4)*ceil(N/4)/64) to match the 4x4-per-thread layout.
(let* ((M 20) (K 24) (N 12)
       (A (nelisp-gpu-gen (* M K) (lambda (i) (* 0.1 (- (mod i 7) 3)))))
       (B (nelisp-gpu-gen (* K N) (lambda (i) (* 0.1 (- (mod i 5) 2)))))
       (C0 (make-vector (* M N) 0.0))
       (ref (photon-tensor-data
             (photon-tensor-matmul (photon-tensor (list M K) A)
                                   (photon-tensor (list K N) B))))
       (got (nth 2 (nelisp-gpu-run 'matmul-reg (list A B C0) (list M K N)
                                   (/ (+ (* (/ (+ M 3) 4) (/ (+ N 3) 4)) 63) 64)))))
  (verify--ck "matmul-reg" (verify--max-err got ref) 1e-3))

(princ "VERIFY-DONE\n")
;;; verify.el ends here
