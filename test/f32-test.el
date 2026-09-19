;;; f32-test.el --- the float32 wire encoding -*- lexical-binding: t -*-

;; Every float that reaches a kernel as buffer data, and every float constant
;; compiled into a shader, goes through `nelisp-gpu--f32-bits'.  It used to
;; mask the exponent with #xff and do nothing else, which is correct only
;; while every value happens to be of order one.  Outside that range it did
;; not clamp, saturate or signal -- it wrapped:
;;
;;     1e-39   encoded to  1.16e+38
;;     3.7e-44 encoded to  4.28e+33
;;     1e+300  encoded to  5.56e-09
;;
;; This is not an exotic input.  A softmax over a large vocabulary puts most
;; of its entries below 1e-38, so a gradient built from one arrived on the GPU
;; as garbage of order 1e33.  That is how it was found: the transposed matmul
;; agreed with its CPU reference to 1e-05 on pseudo-random input and disagreed
;; by 7.7e+30 on a real gradient, with the kernel entirely innocent.
;;
;; No GPU is needed here.  The encoding is arithmetic, and the check is a
;; round trip through the decoder plus the three boundaries that were wrong.

(require 'nelisp-gpu-compile)
(require 'nelisp-gpu-run)

(defvar f32--fail 0)
(defconst f32--min-normal 1.1754943508222875e-38)
(defconst f32--step 1.401298464324817e-45)      ; the subnormal spacing

(defun f32--ck (name ok &optional extra)
  (princ (format "%-52s %s  %s\n" name
                 (if ok "PASS" (progn (setq f32--fail (1+ f32--fail)) "FAIL"))
                 (or extra ""))))

(defun f32--round (x) (nelisp-gpu--bits-f32 (nelisp-gpu--f32-bits x)))

;; Normal magnitudes: a round trip is exact to float32's 24 bits.
(let ((worst 0.0))
  (dolist (x '(1.0 0.5 -3.25 3.14159265358979 1.0e-10 -2.5e7 6.02e23 1.0e-30))
    (let ((r (f32--round x)))
      (setq worst (max worst (/ (abs (- r x)) (abs x))))))
  (f32--ck "normal magnitudes round trip to float32 precision"
           (< worst 1.0e-7) (format "worst rel %.2e" worst)))

;; Subnormals.  Correctness here is absolute, not relative: below the smallest
;; normal there are only 2^23 representable values in total, so 3.7e-44 has
;; about 26 of them beneath it and cannot be relatively accurate.  Asserting a
;; relative bound here would be asserting something float32 cannot do.
(let ((worst 0.0))
  (dolist (x '(1.0e-38 1.0e-39 1.0e-40 3.7e-44 -3.7e-44 1.0e-45 -1.0e-41))
    (let ((r (f32--round x)))
      (setq worst (max worst (/ (abs (- r x)) f32--step)))))
  (f32--ck "subnormals land within half a subnormal step"
           (<= worst 0.5) (format "worst %.2f steps, spacing %.3e" worst f32--step)))

;; The specific regression: small must stay small.  Before the fix each of
;; these came back larger than 1e+17.
(let ((worst 0.0) (val 0.0))
  (dolist (x '(1.0e-39 1.0e-40 1.0e-42 3.7e-44 1.0e-45 1.0e-60 1.0e-300))
    (let ((r (abs (f32--round x))))
      (when (> r worst) (setq worst r val x))))
  (f32--ck "no small magnitude decodes as a large one"
           (< worst f32--min-normal)
           (format "largest was %.3e (from %.0e)" worst val)))

;; Underflow past the last subnormal is zero, which is what float32 does.
(f32--ck "magnitudes below half a subnormal flush to zero"
         (and (= (f32--round 1.0e-60) 0.0) (= (f32--round 5.0e-46) 0.0))
         "1e-60 and 5e-46")

;; Overflow saturates to an infinity rather than wrapping to a small number.
;; An infinity is visible downstream; 5.6e-09 in place of 1e+300 is not.
(let ((big (f32--round 1.0e+300)) (neg (f32--round -1.0e+39)))
  (f32--ck "overflow becomes an infinity, with its sign"
           (and (= big 1.0e+INF) (= neg -1.0e+INF))
           (format "1e300 -> %s, -1e39 -> %s" big neg)))

;; The largest finite float32 must NOT become an infinity -- otherwise the
;; boundary is off by one representable value and ordinary data saturates.
(let ((m (f32--round 3.4028234663852886e38)))
  (f32--ck "the largest finite float32 stays finite"
           (and (not (= m 1.0e+INF)) (> m 3.4e38))
           (format "%.7e" m)))

;; Zero keeps its bit pattern, and a NaN stays a NaN rather than decoding as a
;; number: a kernel that produced one should not look like it produced data.
(f32--ck "zero encodes to zero" (= (nelisp-gpu--f32-bits 0.0) 0))
(f32--ck "NaN survives the round trip" (isnan (f32--round 0.0e+NaN)))

;; And the whole point, end to end: a real softmax tail, the shape of input
;; that exposed this.  Every entry must come back no larger than it went in.
(let* ((n 4096) (v (make-vector n 0.0)) (bad 0) (worst 1.0) (at 0.0))
  (dotimes (i n) (aset v i (exp (- (* 0.05 i)))))   ; down to e^-204 = 1e-89
  (dotimes (i n)
    (let* ((x (aref v i)) (r (f32--round x))
           ;; Rounding to nearest can land just *above* the double, so the
           ;; criterion is a ratio, not "never larger".  A blow-up is orders
           ;; of magnitude; rounding is one part in 2^24.  Below the smallest
           ;; normal the ratio stops meaning anything -- 2.15e-45 has exactly
           ;; two representable neighbours -- so those are judged absolutely,
           ;; the same way the subnormal check above is.
           (ratio (if (>= x f32--min-normal) (/ r x)
                    (if (<= (abs (- r x)) (* 0.5 f32--step)) 1.0 1.0e9))))
      (when (> ratio 1.001) (setq bad (1+ bad))
            (when (> ratio worst) (setq worst ratio at x)))))
  (f32--ck "a softmax tail encodes without a single blow-up"
           (zerop bad)
           (format "%d of %d grew%s" bad n
                   (if (zerop bad) "" (format ", worst ratio %.3e at %.3e" worst at)))))

(princ (format "\n%s: %d failure(s)\n"
               (if (zerop f32--fail) "f32 OK" "f32") f32--fail))
(when (> f32--fail 0) (kill-emacs 1))
