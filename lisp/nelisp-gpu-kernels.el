;;; nelisp-gpu-kernels.el --- standard GPU kernels written in the elisp DSL  -*- lexical-binding: t; -*-

;; A small registry of compute kernels expressed in the nelisp-gpu
;; s-expression DSL (see nelisp-gpu-compile.el).  Each is compiled to
;; SPIR-V and runs on the GPU via the reference hosts.

;;; Code:

(require 'nelisp-gpu-compile)

(defconst nelisp-gpu-kernels
  '((vadd
     . (:buffers (A B C) :push (n) :local-size 64
        :body ((declare i :uint (gid-x))
               (when (< i n)
                 (store (aref C i) (+ (aref A i) (aref B i)))))))
    (matmul
     . (:buffers (A B C) :push (M K N) :local-size 64
        :body ((declare idx :uint (gid-x))
               (when (< idx (* M N))
                 (declare row :uint (/ idx N))
                 (declare col :uint (% idx N))
                 (declare acc :float 0.0)
                 (for (k 0 K)
                   (set acc (+ acc (* (aref A (+ (* row K) k))
                                      (aref B (+ (* k N) col))))))
                 (store (aref C idx) acc)))))
    (softmax
     . (:buffers (A C) :push (M N) :local-size 64
        :body ((declare idx :uint (gid-x))
               (when (< idx M)
                 (declare base :uint (* idx N))
                 (declare m :float (aref A base))
                 (for (j 0 N)
                   (declare v :float (aref A (+ base j)))
                   (when (> v m) (set m v)))
                 (declare s :float 0.0)
                 (for (j2 0 N)
                   (set s (+ s (exp (- (aref A (+ base j2)) m)))))
                 (for (j3 0 N)
                   (store (aref C (+ base j3))
                          (/ (exp (- (aref A (+ base j3)) m)) s)))))))
    (gelu
     . (:buffers (A C) :push (n) :local-size 64
        :body ((declare i :uint (gid-x))
               (when (< i n)
                 (declare x :float (aref A i))
                 (declare inner :float
                          (* 0.7978845608 (+ x (* 0.044715 (* x (* x x))))))
                 (declare e :float (exp (* 2.0 inner)))
                 (declare th :float (/ (- e 1.0) (+ e 1.0)))
                 (store (aref C i) (* 0.5 (* x (+ 1.0 th))))))))
    (layernorm
     . (:buffers (A C) :push (M N) :local-size 64
        :body ((declare idx :uint (gid-x))
               (when (< idx M)
                 (declare base :uint (* idx N))
                 (declare nf :float (float N))
                 (declare mean :float 0.0)
                 (for (j 0 N) (set mean (+ mean (aref A (+ base j)))))
                 (set mean (/ mean nf))
                 (declare var :float 0.0)
                 (for (j2 0 N)
                   (declare d :float (- (aref A (+ base j2)) mean))
                   (set var (+ var (* d d))))
                 (set var (/ var nf))
                 (declare inv :float (/ 1.0 (sqrt (+ var 0.00001))))
                 (for (j3 0 N)
                   (store (aref C (+ base j3))
                          (* (- (aref A (+ base j3)) mean) inv)))))))
    (linear
     . (:buffers (X W B C) :push (M IN OUT) :local-size 64
        :body ((declare idx :uint (gid-x))
               (when (< idx (* M OUT))
                 (declare row :uint (/ idx OUT))
                 (declare o :uint (% idx OUT))
                 (declare acc :float (aref B o))
                 (for (k 0 IN)
                   (set acc (+ acc (* (aref X (+ (* row IN) k))
                                      (aref W (+ (* o IN) k))))))
                 (store (aref C idx) acc)))))
    (matmul-tiled
     . (:buffers (A B C) :push (M K N) :local-size (16 16)
        :shared ((As 256) (Bs 256))
        :body ((declare lr :uint (lid-y))
               (declare lc :uint (lid-x))
               (declare tilesx :uint (/ (+ N 15) 16))
               (declare trow :uint (/ (wgid-x) tilesx))
               (declare tcol :uint (% (wgid-x) tilesx))
               (declare row :uint (+ (* trow 16) lr))
               (declare col :uint (+ (* tcol 16) lc))
               (declare acc :float 0.0)
               (declare nt :uint (/ (+ K 15) 16))
               (for (tt 0 nt)
                 (declare ak :uint (+ (* tt 16) lc))
                 (declare bk :uint (+ (* tt 16) lr))
                 (store-shared As (+ (* lr 16) lc) 0.0)
                 (when (< row M) (when (< ak K)
                   (store-shared As (+ (* lr 16) lc) (aref A (+ (* row K) ak)))))
                 (store-shared Bs (+ (* lr 16) lc) 0.0)
                 (when (< bk K) (when (< col N)
                   (store-shared Bs (+ (* lr 16) lc) (aref B (+ (* bk N) col)))))
                 (barrier)
                 (set acc (+ acc (+ (* (aref-shared As (+ (* lr 16) 0)) (aref-shared Bs (+ (* 0 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 1)) (aref-shared Bs (+ (* 1 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 2)) (aref-shared Bs (+ (* 2 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 3)) (aref-shared Bs (+ (* 3 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 4)) (aref-shared Bs (+ (* 4 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 5)) (aref-shared Bs (+ (* 5 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 6)) (aref-shared Bs (+ (* 6 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 7)) (aref-shared Bs (+ (* 7 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 8)) (aref-shared Bs (+ (* 8 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 9)) (aref-shared Bs (+ (* 9 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 10)) (aref-shared Bs (+ (* 10 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 11)) (aref-shared Bs (+ (* 11 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 12)) (aref-shared Bs (+ (* 12 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 13)) (aref-shared Bs (+ (* 13 16) lc))) (+ (* (aref-shared As (+ (* lr 16) 14)) (aref-shared Bs (+ (* 14 16) lc))) (* (aref-shared As (+ (* lr 16) 15)) (aref-shared Bs (+ (* 15 16) lc))))))))))))))))))))
                 (barrier))
               (when (< row M) (when (< col N)
                 (store (aref C (+ (* row N) col)) acc)))))))
  "Alist of NAME -> kernel spec plist.")

(defun nelisp-gpu-kernel-spec (name)
  (or (cdr (assq name nelisp-gpu-kernels))
      (error "nelisp-gpu: no kernel %S" name)))

(defun nelisp-gpu-write-kernel (name path)
  "Compile kernel NAME (symbol) to SPIR-V at PATH."
  (nelisp-gpu-compile-to-file (nelisp-gpu-kernel-spec name) path))

;; Batch entry: emacs --batch ... -f nelisp-gpu-batch-write NAME PATH
(defun nelisp-gpu-batch-write ()
  (let ((name (intern (nth 0 command-line-args-left)))
        (path (nth 1 command-line-args-left)))
    (setq command-line-args-left nil)   ; consume so Emacs does not visit them
    (nelisp-gpu-write-kernel name path)
    (message "wrote %s -> %s" name path)))

(provide 'nelisp-gpu-kernels)
;;; nelisp-gpu-kernels.el ends here

(push (cons 'matmul-reg '(:buffers (A B C) :push (M K N) :local-size 64 :body ((declare idx :uint (gid-x)) (declare bpr :uint (/ (+ N 3) 4)) (declare brow :uint (* (/ idx bpr) 4)) (declare bcol :uint (* (% idx bpr) 4)) (when (< brow M) (declare acc00 :float 0.0) (declare acc01 :float 0.0) (declare acc02 :float 0.0) (declare acc03 :float 0.0) (declare acc10 :float 0.0) (declare acc11 :float 0.0) (declare acc12 :float 0.0) (declare acc13 :float 0.0) (declare acc20 :float 0.0) (declare acc21 :float 0.0) (declare acc22 :float 0.0) (declare acc23 :float 0.0) (declare acc30 :float 0.0) (declare acc31 :float 0.0) (declare acc32 :float 0.0) (declare acc33 :float 0.0) (for (k 0 K) (declare a0 :float (aref A (+ (* (+ brow 0) K) k))) (declare a1 :float (aref A (+ (* (+ brow 1) K) k))) (declare a2 :float (aref A (+ (* (+ brow 2) K) k))) (declare a3 :float (aref A (+ (* (+ brow 3) K) k))) (declare b0 :float (aref B (+ (* k N) (+ bcol 0)))) (declare b1 :float (aref B (+ (* k N) (+ bcol 1)))) (declare b2 :float (aref B (+ (* k N) (+ bcol 2)))) (declare b3 :float (aref B (+ (* k N) (+ bcol 3)))) (set acc00 (+ acc00 (* a0 b0))) (set acc01 (+ acc01 (* a0 b1))) (set acc02 (+ acc02 (* a0 b2))) (set acc03 (+ acc03 (* a0 b3))) (set acc10 (+ acc10 (* a1 b0))) (set acc11 (+ acc11 (* a1 b1))) (set acc12 (+ acc12 (* a1 b2))) (set acc13 (+ acc13 (* a1 b3))) (set acc20 (+ acc20 (* a2 b0))) (set acc21 (+ acc21 (* a2 b1))) (set acc22 (+ acc22 (* a2 b2))) (set acc23 (+ acc23 (* a2 b3))) (set acc30 (+ acc30 (* a3 b0))) (set acc31 (+ acc31 (* a3 b1))) (set acc32 (+ acc32 (* a3 b2))) (set acc33 (+ acc33 (* a3 b3)))) (store (aref C (+ (* (+ brow 0) N) (+ bcol 0))) acc00) (store (aref C (+ (* (+ brow 0) N) (+ bcol 1))) acc01) (store (aref C (+ (* (+ brow 0) N) (+ bcol 2))) acc02) (store (aref C (+ (* (+ brow 0) N) (+ bcol 3))) acc03) (store (aref C (+ (* (+ brow 1) N) (+ bcol 0))) acc10) (store (aref C (+ (* (+ brow 1) N) (+ bcol 1))) acc11) (store (aref C (+ (* (+ brow 1) N) (+ bcol 2))) acc12) (store (aref C (+ (* (+ brow 1) N) (+ bcol 3))) acc13) (store (aref C (+ (* (+ brow 2) N) (+ bcol 0))) acc20) (store (aref C (+ (* (+ brow 2) N) (+ bcol 1))) acc21) (store (aref C (+ (* (+ brow 2) N) (+ bcol 2))) acc22) (store (aref C (+ (* (+ brow 2) N) (+ bcol 3))) acc23) (store (aref C (+ (* (+ brow 3) N) (+ bcol 0))) acc30) (store (aref C (+ (* (+ brow 3) N) (+ bcol 1))) acc31) (store (aref C (+ (* (+ brow 3) N) (+ bcol 2))) acc32) (store (aref C (+ (* (+ brow 3) N) (+ bcol 3))) acc33))))) nelisp-gpu-kernels)

;; --- glue kernels for attention-block fusion -------------------------
(push (cons 'transpose
            '(:buffers (A C) :push (R Co) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* R Co))
                       (declare i :uint (/ idx Co))
                       (declare j :uint (% idx Co))
                       (store (aref C (+ (* j R) i)) (aref A idx))))))
      nelisp-gpu-kernels)

(push (cons 'scale
            '(:buffers (A S C) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (store (aref C i) (* (aref A i) (aref S 0)))))))
      nelisp-gpu-kernels)

(push (cons 'causal-mask
            '(:buffers (S) :push (Nn) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* Nn Nn))
                       (declare i :uint (/ idx Nn))
                       (declare j :uint (% idx Nn))
                       (when (> j i) (store (aref S idx) -1.0e30))))))
      nelisp-gpu-kernels)

(push (cons 'slice-cols
            '(:buffers (A C) :push (seq dim hd c0) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq hd))
                       (declare i :uint (/ idx hd))
                       (declare j :uint (% idx hd))
                       (store (aref C idx) (aref A (+ (+ (* i dim) c0) j)))))))
      nelisp-gpu-kernels)

(push (cons 'set-cols
            '(:buffers (D S) :push (seq dim hd c0) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq hd))
                       (declare i :uint (/ idx hd))
                       (declare j :uint (% idx hd))
                       (store (aref D (+ (+ (* i dim) c0) j)) (aref S idx))))))
      nelisp-gpu-kernels)

;; A^T B: C[m,n] = sum_k A[k,m]*B[k,n]  (A is K x M, B is K x N).  Lets a linear's
;; weight-gradient dW = g^T x be one dispatch (no separate transpose + buffer).
(push (cons 'matmul-at
            '(:buffers (A B C) :push (M K N) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* M N))
                       (declare m :uint (/ idx N)) (declare nn :uint (% idx N))
                       (declare acc :float 0.0)
                       (for (k 0 K) (set acc (+ acc (* (aref A (+ (* k M) m)) (aref B (+ (* k N) nn))))))
                       (store (aref C idx) acc)))))
      nelisp-gpu-kernels)
;; Fused SwiGLU gate: C[i] = silu(A[i]) * B[i].
(push (cons 'silu-mul
            '(:buffers (A B C) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (declare a :float (aref A i)) (declare sg :float (/ 1.0 (+ 1.0 (exp (- 0.0 a)))))
                       (store (aref C i) (* (* a sg) (aref B i)))))))
      nelisp-gpu-kernels)
;; Fused SwiGLU gate backward: DA = G*B*silu'(A); DB = G*silu(A).
(push (cons 'silu-mul-bwd
            '(:buffers (G A B DA DB) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (declare a :float (aref A i)) (declare sg :float (/ 1.0 (+ 1.0 (exp (- 0.0 a)))))
                       (declare g :float (aref G i))
                       (store (aref DA i) (* (* g (aref B i)) (* sg (+ 1.0 (* a (- 1.0 sg))))))
                       (store (aref DB i) (* g (* a sg)))))))
      nelisp-gpu-kernels)

;; --- fused multi-head attention (all heads in one dispatch each) ------
;; scores S[h,i,j] (laid out (heads*seq) x seq) = (1/sqrt(hd)) * <q_i^h, k_j^kv>
;; with the causal mask folded in (-inf for j>i).  GQA: kv head = h/grp.
(push (cons 'attn-scores
            '(:buffers (Q K S) :push (seq dim heads kvheads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* (* heads seq) seq))
                       (declare h :uint (/ idx (* seq seq))) (declare rem :uint (% idx (* seq seq)))
                       (declare i :uint (/ rem seq)) (declare j :uint (% rem seq))
                       (when (> j i) (store (aref S idx) -1.0e30))
                       (when (< j (+ i 1))
                         (declare kvh :uint (/ h grp))
                         (declare qb :uint (+ (* i dim) (* h hd))) (declare kb :uint (+ (* j kvdim) (* kvh hd)))
                         (declare acc :float 0.0)
                         (for (t 0 hd) (set acc (+ acc (* (aref Q (+ qb t)) (aref K (+ kb t))))))
                         (store (aref S idx) (* acc (/ 1.0 (sqrt (float hd))))))))))
      nelisp-gpu-kernels)
;; context C[i, h*hd+t] (seq x dim) = sum_j P[h,i,j] * V[j, kvh*hd+t].
(push (cons 'attn-context
            '(:buffers (P V C) :push (seq dim heads kvheads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* seq dim))
                       (declare i :uint (/ idx dim)) (declare c :uint (% idx dim))
                       (declare h :uint (/ c hd)) (declare t :uint (% c hd)) (declare kvh :uint (/ h grp))
                       (declare pb :uint (+ (* h (* seq seq)) (* i seq)))
                       (declare acc :float 0.0)
                       (for (j 0 seq) (set acc (+ acc (* (aref P (+ pb j)) (aref V (+ (* j kvdim) (+ (* kvh hd) t)))))))
                       (store (aref C idx) acc)))))
      nelisp-gpu-kernels)
;; dQ[i,h*hd+t] = (1/sqrt(hd)) * sum_j DS[h,i,j] * K[j,kvh*hd+t].
(push (cons 'attn-sc-dq
            '(:buffers (DS K DQ) :push (seq dim heads kvheads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* seq dim))
                       (declare i :uint (/ idx dim)) (declare c :uint (% idx dim))
                       (declare h :uint (/ c hd)) (declare t :uint (% c hd)) (declare kvh :uint (/ h grp))
                       (declare sb :uint (+ (* h (* seq seq)) (* i seq)))
                       (declare acc :float 0.0)
                       (for (j 0 seq) (set acc (+ acc (* (aref DS (+ sb j)) (aref K (+ (* j kvdim) (+ (* kvh hd) t)))))))
                       (store (aref DQ idx) (* acc (/ 1.0 (sqrt (float hd)))))))))
      nelisp-gpu-kernels)
;; dK[j,kvh*hd+t] = (1/sqrt(hd)) * sum_{h in kvh's group} sum_i DS[h,i,j] * Q[i,h*hd+t].
(push (cons 'attn-sc-dk
            '(:buffers (DS Q DK) :push (seq dim heads kvheads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* seq kvdim))
                       (declare j :uint (/ idx kvdim)) (declare rem :uint (% idx kvdim))
                       (declare kvh :uint (/ rem hd)) (declare t :uint (% rem hd)) (declare h0 :uint (* kvh grp))
                       (declare acc :float 0.0)
                       (for (hh 0 grp)
                         (declare h :uint (+ h0 hh)) (declare sb :uint (+ (* h (* seq seq)) j)) (declare qb :uint (+ (* h hd) t))
                         (for (i 0 seq) (set acc (+ acc (* (aref DS (+ sb (* i seq))) (aref Q (+ (* i dim) qb)))))))
                       (store (aref DK idx) (* acc (/ 1.0 (sqrt (float hd)))))))))
      nelisp-gpu-kernels)
;; dP[h,i,j] = sum_t DC[i,h*hd+t] * V[j,kvh*hd+t].
(push (cons 'attn-ctx-dp
            '(:buffers (DC V DP) :push (seq dim heads kvheads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* (* heads seq) seq))
                       (declare h :uint (/ idx (* seq seq))) (declare rem :uint (% idx (* seq seq)))
                       (declare i :uint (/ rem seq)) (declare j :uint (% rem seq)) (declare kvh :uint (/ h grp))
                       (declare cb :uint (+ (* i dim) (* h hd))) (declare vb :uint (+ (* j kvdim) (* kvh hd)))
                       (declare acc :float 0.0)
                       (for (t 0 hd) (set acc (+ acc (* (aref DC (+ cb t)) (aref V (+ vb t))))))
                       (store (aref DP idx) acc)))))
      nelisp-gpu-kernels)
;; dV[j,kvh*hd+t] = sum_{h in kvh's group} sum_i DC[i,h*hd+t] * P[h,i,j].
(push (cons 'attn-ctx-dv
            '(:buffers (DC P DV) :push (seq dim heads kvheads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* seq kvdim))
                       (declare j :uint (/ idx kvdim)) (declare rem :uint (% idx kvdim))
                       (declare kvh :uint (/ rem hd)) (declare t :uint (% rem hd)) (declare h0 :uint (* kvh grp))
                       (declare acc :float 0.0)
                       (for (hh 0 grp)
                         (declare h :uint (+ h0 hh)) (declare pb :uint (+ (* h (* seq seq)) j)) (declare cb :uint (+ (* h hd) t))
                         (for (i 0 seq) (set acc (+ acc (* (aref DC (+ (* i dim) cb)) (aref P (+ pb (* i seq))))))))
                       (store (aref DV idx) acc)))))
      nelisp-gpu-kernels)

;; --- KV-cache incremental decode (single token, runtime position POS[0]) ----
;; RoPE a single row X (1 x cols) at runtime position POS[0], using cos/sin tables
;; CO/SI (max_seq x hd/2) indexed by that position.  SG[0] = +1 (sign).
(push (cons 'decode-rope
            '(:buffers (X CO SI SG POS C) :push (cols heads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ cols heads)) (declare half :uint (/ hd 2))
                     (when (< idx (* heads half))
                       (declare m :uint (% idx half)) (declare h :uint (/ idx half))
                       (declare pos :uint (uint (aref POS 0))) (declare ci :uint (+ (* pos half) m))
                       (declare cc :float (aref CO ci)) (declare ss :float (* (aref SI ci) (aref SG 0)))
                       (declare b0 :uint (* h hd)) (declare i0 :uint (+ b0 (* 2 m))) (declare i1 :uint (+ i0 1))
                       (declare a0 :float (aref X i0)) (declare a1 :float (aref X i1))
                       (store (aref C i0) (- (* a0 cc) (* a1 ss)))
                       (store (aref C i1) (+ (* a0 ss) (* a1 cc)))))))
      nelisp-gpu-kernels)
;; Append a (kvdim) vector SRC into CACHE at row POS[0] (CACHE persists resident).
(push (cons 'cache-append
            '(:buffers (SRC POS CACHE) :push (kvdim) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx kvdim)
                       (declare pos :uint (uint (aref POS 0)))
                       (store (aref CACHE (+ (* pos kvdim) idx)) (aref SRC idx))))))
      nelisp-gpu-kernels)
;; Single-query causal attention over the cache CK/CV (max_seq x kvdim) for
;; positions 0..POS[0], GQA (kv head = h/grp).  C (1 x dim) = context.
(push (cons 'decode-attn
            '(:buffers (Q CK CV POS C) :push (dim heads kvheads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx dim)
                       (declare h :uint (/ idx hd)) (declare t :uint (% idx hd)) (declare kvh :uint (/ h grp))
                       (declare pos :uint (uint (aref POS 0))) (declare len :uint (+ pos 1))
                       (declare c0q :uint (* h hd)) (declare coef :float (/ 1.0 (sqrt (float hd))))
                       (declare mx :float -1.0e30)
                       (for (j 0 len)
                         (declare kb :uint (+ (* j kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare s :float (* acc coef)) (when (> s mx) (set mx s)))
                       (declare z :float 0.0) (declare ctx :float 0.0)
                       (for (j 0 len)
                         (declare kb :uint (+ (* j kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare e :float (exp (- (* acc coef) mx)))
                         (set z (+ z e))
                         (set ctx (+ ctx (* e (aref CV (+ (+ (* j kvdim) (* kvh hd)) t))))))
                       (store (aref C idx) (/ ctx z))))))
      nelisp-gpu-kernels)

;; --- batched (B-sequence) decode: B rows, one shared position POS[0] -------
;; RoPE B rows X (B x cols) all at position POS[0].
(push (cons 'decode-rope-b
            '(:buffers (X CO SI SG POS C) :push (B cols heads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ cols heads)) (declare half :uint (/ hd 2)) (declare rh :uint (* heads half))
                     (when (< idx (* B rh))
                       (declare bb :uint (/ idx rh)) (declare rem :uint (% idx rh))
                       (declare m :uint (% rem half)) (declare h :uint (/ rem half))
                       (declare pos :uint (uint (aref POS 0))) (declare ci :uint (+ (* pos half) m))
                       (declare cc :float (aref CO ci)) (declare ss :float (* (aref SI ci) (aref SG 0)))
                       (declare b0 :uint (+ (* bb cols) (* h hd))) (declare i0 :uint (+ b0 (* 2 m))) (declare i1 :uint (+ i0 1))
                       (declare a0 :float (aref X i0)) (declare a1 :float (aref X i1))
                       (store (aref C i0) (- (* a0 cc) (* a1 ss)))
                       (store (aref C i1) (+ (* a0 ss) (* a1 cc)))))))
      nelisp-gpu-kernels)
;; Append B rows SRC (B x kvdim) into B per-sequence caches at row POS[0].
;; CACHE is (B x maxseq x kvdim); sequence b's row pos at (b*maxseq+pos)*kvdim.
(push (cons 'cache-append-b
            '(:buffers (SRC POS CACHE) :push (B kvdim maxseq) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* B kvdim))
                       (declare bb :uint (/ idx kvdim)) (declare t :uint (% idx kvdim))
                       (declare pos :uint (uint (aref POS 0)))
                       (store (aref CACHE (+ (* (+ (* bb maxseq) pos) kvdim) t)) (aref SRC idx))))))
      nelisp-gpu-kernels)
;; Per-sequence single-query attention: C (B x dim), sequence b attends its own
;; cache CK/CV (B x maxseq x kvdim) over positions 0..POS[0].
(push (cons 'decode-attn-b
            '(:buffers (Q CK CV POS C) :push (B dim heads kvheads maxseq) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* B dim))
                       (declare bb :uint (/ idx dim)) (declare c :uint (% idx dim))
                       (declare h :uint (/ c hd)) (declare t :uint (% c hd)) (declare kvh :uint (/ h grp))
                       (declare pos :uint (uint (aref POS 0))) (declare len :uint (+ pos 1))
                       (declare cbase :uint (* (* bb maxseq) kvdim)) (declare c0q :uint (+ (* bb dim) (* h hd)))
                       (declare coef :float (/ 1.0 (sqrt (float hd)))) (declare mx :float -1.0e30)
                       (for (j 0 len)
                         (declare kb :uint (+ cbase (+ (* j kvdim) (* kvh hd)))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare s :float (* acc coef)) (when (> s mx) (set mx s)))
                       (declare z :float 0.0) (declare ctx :float 0.0)
                       (for (j 0 len)
                         (declare kb :uint (+ cbase (+ (* j kvdim) (* kvh hd)))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare e :float (exp (- (* acc coef) mx)))
                         (set z (+ z e))
                         (set ctx (+ ctx (* e (aref CV (+ kb t))))))
                       (store (aref C idx) (/ ctx z))))))
      nelisp-gpu-kernels)

;; --- StreamingLLM bounded decode (attention sink + rolling window) ---------
;; Cache holds NSINK permanent sink slots + WIN rolling slots = cap entries.
;; Slot for stream position pos: pos<nsink ? pos : nsink + ((pos-nsink) mod win).
;; Append a RAW (un-rotated) (kvdim) vector SRC at the ring slot for POS[0].
(push (cons 'cache-append-ring
            '(:buffers (SRC POS CACHE) :push (kvdim nsink win) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx kvdim)
                       (declare pos :uint (uint (aref POS 0)))
                       (declare slot :uint pos)
                       (when (>= pos nsink) (set slot (+ nsink (% (- pos nsink) win))))
                       (store (aref CACHE (+ (* slot kvdim) idx)) (aref SRC idx))))))
      nelisp-gpu-kernels)
;; Single-query attention with cache-relative RoPE.  Q/CK are RAW; the query is
;; rotated by its cache-relative position qcrel and each cached key by its
;; cache-relative rank r (CO/SI tables, >= cap rows); V (CV) is never rotated.
;; All of len/start/woff/qcrel are derived in-kernel from POS[0] + nsink/win.
(push (cons 'decode-attn-stream
            '(:buffers (Q CK CV POS CO SI C) :push (dim heads kvheads nsink win) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads))
                     (declare kvdim :uint (* kvheads hd)) (declare half :uint (/ hd 2))
                     (when (< idx dim)
                       (declare h :uint (/ idx hd)) (declare t :uint (% idx hd)) (declare kvh :uint (/ h grp))
                       (declare pos :uint (uint (aref POS 0))) (declare cap :uint (+ nsink win))
                       (declare len :uint (+ pos 1)) (when (> len cap) (set len cap))
                       (declare start :uint nsink) (when (>= pos cap) (set start (+ (- pos win) 1)))
                       (declare woff :uint (% (- start nsink) win))
                       (declare qcrel :uint pos) (when (>= qcrel cap) (set qcrel (- cap 1)))
                       (declare c0q :uint (* h hd)) (declare coef :float (/ 1.0 (sqrt (float hd))))
                       (declare mx :float -1.0e30)
                       (for (r 0 len)
                         (declare slot :uint r) (when (>= r nsink) (set slot (+ nsink (% (+ woff (- r nsink)) win))))
                         (declare kb :uint (+ (* slot kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (m 0 half)
                           (declare ci :uint (+ (* r half) m)) (declare cc :float (aref CO ci)) (declare ss :float (aref SI ci))
                           (declare k0 :float (aref CK (+ kb (* 2 m)))) (declare k1 :float (aref CK (+ kb (+ (* 2 m) 1))))
                           (declare kr0 :float (- (* k0 cc) (* k1 ss))) (declare kr1 :float (+ (* k0 ss) (* k1 cc)))
                           (declare qci :uint (+ (* qcrel half) m)) (declare qcc :float (aref CO qci)) (declare qss :float (aref SI qci))
                           (declare q0 :float (aref Q (+ c0q (* 2 m)))) (declare q1 :float (aref Q (+ c0q (+ (* 2 m) 1))))
                           (declare qr0 :float (- (* q0 qcc) (* q1 qss))) (declare qr1 :float (+ (* q0 qss) (* q1 qcc)))
                           (set acc (+ acc (+ (* qr0 kr0) (* qr1 kr1)))))
                         (declare s :float (* acc coef)) (when (> s mx) (set mx s)))
                       (declare z :float 0.0) (declare ctx :float 0.0)
                       (for (r 0 len)
                         (declare slot :uint r) (when (>= r nsink) (set slot (+ nsink (% (+ woff (- r nsink)) win))))
                         (declare kb :uint (+ (* slot kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (m 0 half)
                           (declare ci :uint (+ (* r half) m)) (declare cc :float (aref CO ci)) (declare ss :float (aref SI ci))
                           (declare k0 :float (aref CK (+ kb (* 2 m)))) (declare k1 :float (aref CK (+ kb (+ (* 2 m) 1))))
                           (declare kr0 :float (- (* k0 cc) (* k1 ss))) (declare kr1 :float (+ (* k0 ss) (* k1 cc)))
                           (declare qci :uint (+ (* qcrel half) m)) (declare qcc :float (aref CO qci)) (declare qss :float (aref SI qci))
                           (declare q0 :float (aref Q (+ c0q (* 2 m)))) (declare q1 :float (aref Q (+ c0q (+ (* 2 m) 1))))
                           (declare qr0 :float (- (* q0 qcc) (* q1 qss))) (declare qr1 :float (+ (* q0 qss) (* q1 qcc)))
                           (set acc (+ acc (+ (* qr0 kr0) (* qr1 kr1)))))
                         (declare e :float (exp (- (* acc coef) mx))) (set z (+ z e))
                         (declare vidx :uint (+ (* slot kvdim) (+ (* kvh hd) t)))
                         (set ctx (+ ctx (* e (aref CV vidx)))))
                       (store (aref C idx) (/ ctx z))))))
      nelisp-gpu-kernels)

;; --- on-device training kernels (fused forward+backward+SGD) ----------
;; C[i] = (A[i] - B[i]) * S[0].  Used for the scaled MSE gradient dy.
(push (cons 'sub-scale
            '(:buffers (A B S C) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (store (aref C i) (* (- (aref A i) (aref B i)) (aref S 0)))))))
      nelisp-gpu-kernels)

;; Column sum: D[o] = sum over rows r<M of G[r*N+o].  Bias gradient.
(push (cons 'colsum
            '(:buffers (G D) :push (M N) :local-size 64
              :body ((declare o :uint (gid-x))
                     (when (< o N)
                       (declare acc :float 0.0)
                       (for (r 0 M) (set acc (+ acc (aref G (+ (* r N) o)))))
                       (store (aref D o) acc)))))
      nelisp-gpu-kernels)

;; GELU backward: D[i] = G[i] * gelu'(H[i])  (tanh-approx derivative).
(push (cons 'gelu-bwd
            '(:buffers (G H D) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (declare x :float (aref H i))
                       (declare inner :float (* 0.7978845608 (+ x (* 0.044715 (* x (* x x))))))
                       (declare e :float (exp (* 2.0 inner)))
                       (declare th :float (/ (- e 1.0) (+ e 1.0)))
                       (declare sech2 :float (- 1.0 (* th th)))
                       (declare dudx :float (* 0.7978845608 (+ 1.0 (* 0.134145 (* x x)))))
                       (declare gp :float (+ (* 0.5 (+ 1.0 th)) (* 0.5 (* x (* sech2 dudx)))))
                       (store (aref D i) (* (aref G i) gp))))))
      nelisp-gpu-kernels)

;; In-place SGD with a global-norm clip scale S[0]: W[i] -= L[0]*S[0]*G[i].
(push (cons 'sgd
            '(:buffers (W G L S) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (store (aref W i) (- (aref W i) (* (* (aref L 0) (aref S 0)) (aref G i))))))))
      nelisp-gpu-kernels)
;; Accumulate the sum of squares of G into GSQ[0] (single thread; one dispatch per
;; parameter, serialised by the batch barriers -> global grad sum-of-squares).
(push (cons 'sumsq-acc
            '(:buffers (G GSQ) :push (n) :local-size 64
              :body ((declare t :uint (gid-x))
                     (when (< t 1)
                       (declare acc :float 0.0)
                       (for (i 0 n) (declare g :float (aref G i)) (set acc (+ acc (* g g))))
                       (store (aref GSQ 0) (+ (aref GSQ 0) acc))))))
      nelisp-gpu-kernels)
;; Global-norm clip scale: SCALE[0] = min(1, CFG[0] / sqrt(GSQ[0] + 1e-12)).
(push (cons 'clip-scale
            '(:buffers (GSQ CFG SCALE) :push (n) :local-size 64
              :body ((declare t :uint (gid-x))
                     (when (< t 1)
                       (declare s :float (/ (aref CFG 0) (sqrt (+ (aref GSQ 0) 1.0e-12))))
                       (store (aref SCALE 0) 1.0)
                       (when (< s 1.0) (store (aref SCALE 0) s))))))
      nelisp-gpu-kernels)

;; --- BitNet b1.58 ternary weight quantization (QAT, Phase A) ----------
;; ACC[0] += sum |G[i]|  (serial single-thread reduction; mean|W| numerator).
(push (cons 'absmean-acc
            '(:buffers (G ACC) :push (n) :local-size 64
              :body ((declare t :uint (gid-x))
                     (when (< t 1)
                       (declare acc :float 0.0)
                       (for (i 0 n)
                         (declare g :float (aref G i)) (declare a :float g)
                         (when (< g 0.0) (set a (- 0.0 g)))
                         (set acc (+ acc a)))
                       (store (aref ACC 0) (+ (aref ACC 0) acc))))))
      nelisp-gpu-kernels)
;; Per-row int8 activation quant (one thread per row): gamma = max|row|/127,
;; XQ = gamma * clip(round(X/gamma), -127, 127).  round via uint(|q|+0.5) truncation.
(push (cons 'quant-act
            '(:buffers (X XQ) :push (seq cols) :local-size 64
              :body ((declare row :uint (gid-x))
                     (when (< row seq)
                       (declare base :uint (* row cols)) (declare amax :float 0.0)
                       (for (c 0 cols)
                         (declare v :float (aref X (+ base c))) (declare av :float v)
                         (when (< v 0.0) (set av (- 0.0 v))) (when (> av amax) (set amax av)))
                       (declare gamma :float (/ amax 127.0))
                       (for (c 0 cols)
                         (declare v :float (aref X (+ base c))) (declare xq :float 0.0)
                         (when (> gamma 0.0)
                           (declare q :float (/ v gamma)) (declare sgn :float 1.0) (declare aq :float q)
                           (when (< q 0.0) (set sgn -1.0) (set aq (- 0.0 q)))
                           (declare ri :uint (uint (+ aq 0.5))) (declare rf :float (float ri))
                           (when (> rf 127.0) (set rf 127.0))
                           (set xq (* gamma (* sgn rf))))
                         (store (aref XQ (+ base c)) xq))))))
      nelisp-gpu-kernels)
;; Ternary weight: WQ[i] = beta * round(clip(W[i]/beta, -1, 1)), beta = S[0]/n.
;; round+clip to {-1,0,1} is just two thresholds, so no round/floor builtin needed.
(push (cons 'quant-w
            '(:buffers (W S WQ) :push (n) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx n)
                       (declare beta :float (/ (aref S 0) (float n)))
                       (declare r :float 0.0)
                       (when (> beta 0.0)
                         (declare q :float (/ (aref W idx) beta))
                         (when (>= q 0.5) (set r 1.0))
                         (when (<= q -0.5) (set r -1.0)))
                       (store (aref WQ idx) (* beta r))))))
      nelisp-gpu-kernels)

;; --- PagedAttention feasibility spike: table-indirected gather/scatter ------
;; The load-bearing unknown for paged KV (docs/design/03) is whether the compiler
;; can emit a DATA-DEPENDENT dynamic index: read a physical block id from a TABLE
;; buffer and use it to index a POOL buffer -- POOL[TABLE[blk]*bs + off].  These
;; two kernels isolate exactly that double indirection (load and store).
;; OUT[blk*bs+off] = POOL[ uint(TABLE[blk]) * bs + off ]   (gather / attn read).
(push (cons 'gather-spike
            '(:buffers (TABLE POOL OUT) :push (n bs) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* n bs))
                       (declare blk :uint (/ idx bs)) (declare off :uint (% idx bs))
                       (declare phys :uint (uint (aref TABLE blk)))
                       (store (aref OUT idx) (aref POOL (+ (* phys bs) off)))))))
      nelisp-gpu-kernels)
;; POOL[ uint(TABLE[blk]) * bs + off ] = SRC[blk*bs+off]   (scatter / cache append).
(push (cons 'scatter-spike
            '(:buffers (TABLE SRC POOL) :push (n bs) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* n bs))
                       (declare blk :uint (/ idx bs)) (declare off :uint (% idx bs))
                       (declare phys :uint (uint (aref TABLE blk)))
                       (store (aref POOL (+ (* phys bs) off)) (aref SRC idx))))))
      nelisp-gpu-kernels)

;; --- DP4A feasibility spike: hardware int8 4-lane dot product ---------------
;; Each uint32 holds four int8 lanes; we carry it as two f32 halves (each a
;; 0..65535 integer value, exact in f32) and rebuild it in-kernel as
;; lo + hi*65536, then OpSDot (signed 4x8 packed) gives the int dot product.
;; This de-risks the compiler's OpSDot + DotProduct capability + signed-int path.
(push (cons 'dp4a-dot
            '(:buffers (ALO AHI BLO BHI OUT) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (declare apk :uint (+ (uint (aref ALO i)) (* (uint (aref AHI i)) 65536)))
                       (declare bpk :uint (+ (uint (aref BLO i)) (* (uint (aref BHI i)) 65536)))
                       (declare d :int (sdot apk bpk))
                       (store (aref OUT i) (float d))))))
      nelisp-gpu-kernels)

;; DP4A int8 matmul: Y(seq x out) = BIAS + BETA[0]*GAMMA[s] * sum_lanes(int8_a*int8_w).
;; Activations int8 (per-row GAMMA) and ternary weights int8, each carried as 4
;; lanes/group in two f32 halves (lo+hi*65536); the inner loop is in/4 OpSDot
;; calls (hardware DP4A), 4 MACs per instruction -- the compute win over f32.
(push (cons 'bitlinear-dp4a
            '(:buffers (ALO AHI WLO WHI BIAS BETA GAMMA Y) :push (seq out ng) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq out))
                       (declare s :uint (/ idx out)) (declare o :uint (% idx out))
                       (declare ab :uint (* s ng)) (declare wb :uint (* o ng)) (declare acc :int 0)
                       (for (g 0 ng)
                         (declare apk :uint (+ (uint (aref ALO (+ ab g))) (* (uint (aref AHI (+ ab g))) 65536)))
                         (declare wpk :uint (+ (uint (aref WLO (+ wb g))) (* (uint (aref WHI (+ wb g))) 65536)))
                         (set acc (+ acc (sdot apk wpk))))
                       (store (aref Y idx) (+ (aref BIAS o) (* (* (aref BETA 0) (aref GAMMA s)) (float acc))))))))
      nelisp-gpu-kernels)

;; DP4A with 1 f32 per 4 lanes (bitcast): four int8 lanes are packed into the 32
;; bits of one f32 (uploaded verbatim); the kernel bit-casts it back to a uint32
;; and OpSDots.  1 byte/weight (4x less than f32) AND hardware DP4A -- memory and
;; compute together (vs the 2-f32 carry's 2 bytes/weight).
(push (cons 'bitlinear-dp4a-1f
            '(:buffers (AP WP BIAS BETA GAMMA Y) :push (seq out ng) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq out))
                       (declare s :uint (/ idx out)) (declare o :uint (% idx out))
                       (declare ab :uint (* s ng)) (declare wb :uint (* o ng)) (declare acc :int 0)
                       (for (g 0 ng)
                         (declare apk :uint (bitcast-u (aref AP (+ ab g))))
                         (declare wpk :uint (bitcast-u (aref WP (+ wb g))))
                         (set acc (+ acc (sdot apk wpk))))
                       (store (aref Y idx) (+ (aref BIAS o) (* (* (aref BETA 0) (aref GAMMA s)) (float acc))))))))
      nelisp-gpu-kernels)

;; Per-output-row weight scales.  A sibling of `bitlinear-dp4a-1f' rather than a
;; re-index of it: that kernel's BETA is read as BETA[0] and its callers in
;; nl-llm-bitnet.el pass a ONE-element buffer, so indexing it by the output row
;; would read past the end of the ternary path's uploads.  Imported weights are
;; quantized per row -- a single scale is set by the largest row and wastes most
;; of the int8 range on the others -- so they need BETA of length `out'.
(push (cons 'bitlinear-dp4a-rows
            '(:buffers (AP WP BIAS BETA GAMMA Y) :push (seq out ng) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq out))
                       (declare s :uint (/ idx out)) (declare o :uint (% idx out))
                       (declare ab :uint (* s ng)) (declare wb :uint (* o ng)) (declare acc :int 0)
                       (for (g 0 ng)
                         (declare apk :uint (bitcast-u (aref AP (+ ab g))))
                         (declare wpk :uint (bitcast-u (aref WP (+ wb g))))
                         (set acc (+ acc (sdot apk wpk))))
                       (store (aref Y idx) (+ (aref BIAS o) (* (* (aref BETA o) (aref GAMMA s)) (float acc))))))))
      nelisp-gpu-kernels)

;; One position of the gated delta rule, for every head at once.
;;
;;   S  <- g_t S
;;   m  <- S' k~
;;   d  <- beta_t (v - m)
;;   S  <- S + k~ (x) d
;;   o  <- S' q~
;;
;; The recurrence is sequential in t and completely parallel in everything
;; else: for a fixed head and a fixed output dimension j, the whole update
;; touches only column j of that head's state.  So a thread owns a column,
;; there is no sharing between threads within a position, and the caller
;; dispatches this once per position with T as a push constant while the
;; state stays in a tmp slot across the batch.
;;
;; S is [head][i][j], Q and K are [t][head][i], V and OUT are [t][head][j].
;; Reading and writing S in the same thread is safe because the column a
;; thread touches is its own.
(push (cons 'gdn-step
            '(:buffers (S Q K V G B OUT) :push (nv dk dv t) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* nv dv))
                       (declare h :uint (/ idx dv))
                       (declare j :uint (% idx dv))
                       (declare sb :uint (+ (* h (* dk dv)) j))
                       (declare qb :uint (* (+ (* t nv) h) dk))
                       (declare vb :uint (+ (* (+ (* t nv) h) dv) j))
                       (declare gv :float (aref G (+ (* t nv) h)))
                       (declare bv :float (aref B (+ (* t nv) h)))
                       (declare mem :float 0.0)
                       (for (i 0 dk)
                         (declare p :uint (+ sb (* i dv)))
                         (declare sv :float (* (aref S p) gv))
                         (store (aref S p) sv)
                         (set mem (+ mem (* sv (aref K (+ qb i))))))
                       (declare d :float (* bv (- (aref V vb) mem)))
                       (declare o :float 0.0)
                       (for (i 0 dk)
                         (declare p :uint (+ sb (* i dv)))
                         (declare sv :float (+ (aref S p) (* (aref K (+ qb i)) d)))
                         (store (aref S p) sv)
                         (set o (+ o (* sv (aref Q (+ qb i))))))
                       (store (aref OUT vb) o)))))
      nelisp-gpu-kernels)

;; Ternary weights: sixteen two-bit fields per word, one scale per 128 columns.
;;
;; No DP4A here, and that is the point rather than an omission.  The weight is
;; -1, 0 or +1, so the product is a float add or subtract and there is nothing
;; for an int8 dot product to accelerate; taking the activation as float
;; instead of packing it to int8 costs an instruction per weight and removes
;; the activation quantization entirely, which is the only lossy step left in
;; this path once the weights stop being requantized.  Against the int8 kernel
;; it reads a quarter of the bytes.
;;
;; The scale is per 128-column block, so the accumulator flushes at block
;; boundaries -- eight words of sixteen fields each.  `sh' walks the fields by
;; multiplying by four, which keeps the shift out of the inner loop.
(push (cons 'ternary-rows
            '(:buffers (X WP BIAS BETA Y) :push (seq out cols nblk wng)
              :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq out))
                       (declare s :uint (/ idx out))
                       (declare o :uint (% idx out))
                       (declare xb :uint (* s cols))
                       (declare wb :uint (* o wng))
                       (declare acc :float 0.0)
                       (for (b 0 nblk)
                         (declare bacc :float 0.0)
                         (for (w 0 8)
                           (declare wi :uint (+ (* b 8) w))
                           (when (< wi wng)
                             (declare wpk :uint (bitcast-u (aref WP (+ wb wi))))
                             ;; Field f lives in bits 2f..2f+1, so shifting it
                             ;; up to the sign bit and back down arithmetically
                             ;; both selects and sign-extends it: 00 -> 0,
                             ;; 01 -> +1, 11 -> -1.  The obvious spelling --
                             ;; divide by a running power of four -- costs a
                             ;; 32-bit OpUDiv with a variable divisor per field,
                             ;; and there are COLS of them per output element.
                             (declare sh :uint 30)
                             (for (f 0 16)
                               (declare i :uint (+ (* wi 16) f))
                               (when (< i cols)
                                 (declare sv :float
                                          (float (>>s (bitcast-i (<< wpk sh)) 30)))
                                 (set bacc (+ bacc (* sv (aref X (+ xb i))))))
                               (set sh (- sh 2)))))
                         (set acc (+ acc (* (aref BETA (+ (* o nblk) b)) bacc))))
                       (store (aref Y idx) (+ (aref BIAS o) acc))))))
      nelisp-gpu-kernels)

;; The transpose of `ternary-rows': X = W^T . G, with the scale read per
;; (output row, input block) rather than per output row.
(push (cons 'ternary-rows-t
            '(:buffers (WP BETA G X) :push (out cols nblk wng seq)
              :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq cols))
                       (declare p :uint (/ idx cols))
                       (declare i :uint (% idx cols))
                       (declare b :uint (/ i 128))
                       (declare wi :uint (/ i 16))
                       (declare fd :uint (% i 16))
                       ;; Was a loop of FD multiplications to build a divisor,
                       ;; per thread, before the first weight was even read.
                       (declare sh :uint (- 30 (* 2 fd)))
                       (declare gb :uint (* p out))
                       (declare acc :float 0.0)
                       (for (o 0 out)
                         (declare wpk :uint (bitcast-u (aref WP (+ (* o wng) wi))))
                         (declare sv :float
                                  (float (>>s (bitcast-i (<< wpk sh)) 30)))
                         (set acc (+ acc (* sv (* (aref BETA (+ (* o nblk) b))
                                                  (aref G (+ gb o)))))))
                       (store (aref X idx) acc)))))
      nelisp-gpu-kernels)

;; The transpose of `bitlinear-dp4a-rows': X = W^T . (BETA * G), for a weight
;; stored row-major with four int8 lanes per word and a scale per output row.
;; This is the gradient of a linear with respect to its input, which a frozen
;; quantized base still has to supply -- "frozen" means untrained, not unused.
;;
;; One thread per input column, which is the awkward direction: column i lives
;; in word i/4, lane i%4 of every row, so each thread strides down the weight
;; instead of along it.  DP4A does not apply, because BETA varies per row and
;; multiplies each term, so the accumulation is float rather than int32.
;;
;; The lane is extracted with integer division and remainder -- the DSL has no
;; shifts -- and sign-extended without a branch: (b + 128) % 256 - 128 maps
;; 0..127 to itself and 128..255 to -128..-1.
;; SEQ gradients at once, laid out as SEQ x OUT in G and SEQ x COLS in X.  The
;; weight is read once per thread either way, so a batch of positions costs one
;; dispatch instead of SEQ of them -- which matters because the caller pays a
;; round trip and an Elisp-side float encoding of G per call, not just a kernel.
;; SEQ = 1 is exactly the old kernel.
(push (cons 'dp4a-rows-t
            '(:buffers (WP BETA G X) :push (out cols ng seq) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq cols))
                       (declare p :uint (/ idx cols))
                       (declare i :uint (% idx cols))
                       (declare word :uint (/ i 4))
                       (declare lane :uint (% i 4))
                       (declare sh :uint 1)
                       (for (l 0 lane) (set sh (* sh 256)))
                       (declare gb :uint (* p out))
                       (declare acc :float 0.0)
                       (for (o 0 out)
                         (declare wpk :uint (bitcast-u (aref WP (+ (* o ng) word))))
                         (declare b :uint (% (/ wpk sh) 256))
                         (declare sv :float (- (float (% (+ b 128) 256)) 128.0))
                         (set acc (+ acc (* sv (* (aref BETA o) (aref G (+ gb o)))))))
                       (store (aref X idx) acc)))))
      nelisp-gpu-kernels)

;; C = A + B, elementwise.  There is a hand-built `vadd' SPIR-V module in
;; nelisp-gpu-spirv.el, but a residual connection inside a fused block wants a
;; kernel whose buffer order and push constants are the same shape as every
;; other kernel here, rather than one whose ABI has to be looked up.
(push (cons 'add2
            '(:buffers (A B C) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (store (aref C i) (+ (aref A i) (aref B i)))))))
      nelisp-gpu-kernels)

;; RMSNorm applied per attention head, which is Qwen3's QK-norm: q and k are
;; normalised within each head, with a shared HD-wide gain, BEFORE the rotation.
;; One thread per (position, head).  EPS is 1e-6, matching the only value the
;; callers use; it is not a push constant because push constants are uint32 and
;; a float one would have to be smuggled through a bitcast for no gain.
(push (cons 'rmsnorm-heads
            '(:buffers (X G Y) :push (seq nheads hd) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq nheads))
                       (declare base :uint (* idx hd))
                       (declare ss :float 0.0)
                       (for (i 0 hd)
                         (declare v :float (aref X (+ base i)))
                         (set ss (+ ss (* v v))))
                       (declare inv :float
                                (/ 1.0 (sqrt (+ (/ ss (float hd)) 0.000001))))
                       (for (j 0 hd)
                         (store (aref Y (+ base j))
                                (* (* (aref X (+ base j)) inv) (aref G j))))))))
      nelisp-gpu-kernels)

;; Half-split rotary embedding -- the GPT-NeoX convention Qwen3 and Llama use,
;; pairing element i with i + HD/2.  NOT `rope-apply', which pairs adjacent
;; elements; the two disagree substantially (an 8-wide head at position 3
;; differs by 5.8) and neither complains, because the vector is the right
;; length either way.  That mismatch is one of the four silent convention
;; errors this project has already paid for once.
;;
;; One thread per (position, head, pair).  Reads X and writes Y, so the two
;; must be different buffers: the rotation needs both halves of the original.
;; SGN is the sine's sign, as a float bit pattern: +1 rotates, -1 rotates back.
;; The inverse of (a0 c - a1 s, a1 c + a0 s) is (a0 c + a1 s, a1 c - a0 s), so
;; the backward is this kernel with the sine negated and nothing else -- which
;; is worth having as one kernel rather than two that could drift apart.
(push (cons 'rope-half
            '(:buffers (X Y) :push (seq nheads hd rbase sgn) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare half :uint (/ hd 2))
                     (when (< idx (* (* seq nheads) half))
                       (declare m :uint (% idx half))
                       (declare tmp :uint (/ idx half))
                       (declare h :uint (% tmp nheads))
                       (declare p :uint (/ tmp nheads))
                       (declare b0 :uint (+ (* p (* nheads hd)) (* h hd)))
                       (declare ex :float (/ (* 2.0 (float m)) (float hd)))
                       ;; RBASE arrives as the float's bit pattern, not as an
                       ;; integer: push constants are uint32 and a rope base is
                       ;; a float in the config (1000000.0).  Truncating it
                       ;; would work for every base anyone uses and quietly
                       ;; stop working for one that is not integral.
                       (declare theta :float
                                (/ (float p) (pow (bitcast-f rbase) ex)))
                       (declare c :float (cos theta))
                       (declare s :float (* (bitcast-f sgn) (sin theta)))
                       (declare a0 :float (aref X (+ b0 m)))
                       (declare a1 :float (aref X (+ b0 (+ m half))))
                       (store (aref Y (+ b0 m)) (- (* a0 c) (* a1 s)))
                       (store (aref Y (+ b0 (+ m half)))
                              (+ (* a1 c) (* a0 s)))))))
      nelisp-gpu-kernels)

;; Quantize SEQ rows to int8 and pack four lanes per word, plus the per-row
;; scale -- the input side of `bitlinear-dp4a-rows' done on the device.
;;
;; This is what lets a tensor that is already on the GPU feed an int8 linear.
;; Without it an f32 intermediate has to come back to Elisp to be packed and go
;; up again, and that round trip is the thing worth avoiding: the boundary
;; costs about 3 microseconds a float out and 1.2 back, which is more than the
;; arithmetic of most of a block.
;;
;; One thread per row.  The rounding is `uint(|q| + 0.5)' with a clamp at 127,
;; matching `nl-llm-wgpu-pack-act' exactly rather than approximately -- these
;; two produce the same bytes or the GPU and CPU paths stop being comparable.
;; Lanes are assembled with multiplies because the DSL has no shifts, and the
;; word is bitcast into the float buffer on the way out.
(push (cons 'pack-act-rows
            '(:buffers (X XQ GAMMA) :push (seq cols ng) :local-size 64
              :body ((declare row :uint (gid-x))
                     (when (< row seq)
                       (declare base :uint (* row cols))
                       (declare amax :float 0.0)
                       (for (c 0 cols)
                         (declare v :float (aref X (+ base c)))
                         (declare av :float v)
                         (when (< v 0.0) (set av (- 0.0 v)))
                         (when (> av amax) (set amax av)))
                       (declare gamma :float (/ amax 127.0))
                       (store (aref GAMMA row) gamma)
                       (declare wb :uint (* row ng))
                       (for (w 0 ng)
                         (declare word :uint 0)
                         (declare sh :uint 1)
                         (for (l 0 4)
                           (declare c :uint (+ (* w 4) l))
                           (declare lane :uint 0)
                           (when (< c cols)
                             (declare v :float (aref X (+ base c)))
                             (declare q :float 0.0)
                             (when (> gamma 0.0) (set q (/ v gamma)))
                             (declare neg :uint 0)
                             (declare aq :float q)
                             (when (< q 0.0) (set neg 1) (set aq (- 0.0 q)))
                             (declare ri :uint (uint (+ aq 0.5)))
                             (when (> ri 127) (set ri 127))
                             (set lane ri)
                             (when (= neg 1) (set lane (% (- 256 ri) 256))))
                           (set word (+ word (* lane sh)))
                           (set sh (* sh 256)))
                         (store (aref XQ (+ wb w)) (bitcast-f word)))))))
      nelisp-gpu-kernels)

;; Causal grouped-query attention, one thread per (head, query position).
;; Q is SEQ x HEADS*HD, K and V are SEQ x KVHEADS*HD, CTX is SEQ x HEADS*HD,
;; all with RoPE and QK-norm already applied -- this is only the attention.
;;
;; Three passes over the keys rather than two, because a thread cannot hold a
;; row of scores: their length is SEQ, which is a push constant, and the DSL
;; has no runtime-sized private array.  So the score for (i,j) is recomputed
;; for the maximum, for the sum, and again while accumulating.  That is four
;; dot products of length HD per key instead of two, which is the right trade
;; on a GPU and the wrong one in Elisp -- where this loop costs 311s a step at
;; seq 48 precisely because it is quadratic and interpreted.
;;
;; The accumulation goes through CTX itself instead of a register array, for
;; the same reason.  Each thread owns one (i, head) slice of it, so there is
;; no sharing and no barrier.
(push (cons 'attn-causal-gqa
            '(:buffers (Q K V CTX) :push (seq heads kvheads hd) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* heads seq))
                       (declare h :uint (/ idx seq))
                       (declare i :uint (% idx seq))
                       (declare qdim :uint (* heads hd))
                       (declare kvdim :uint (* kvheads hd))
                       (declare grp :uint (/ heads kvheads))
                       (declare qb :uint (+ (* i qdim) (* h hd)))
                       (declare kc :uint (* (/ h grp) hd))
                       (declare scale :float (/ 1.0 (sqrt (float hd))))
                       (declare n :uint (+ i 1))
                       ;; pass 1: the maximum, so the exponential cannot overflow
                       (declare mx :float -1.0e30)
                       (for (j 0 n)
                         (declare kb :uint (+ (* j kvdim) kc))
                         (declare acc :float 0.0)
                         (for (t 0 hd)
                           (set acc (+ acc (* (aref Q (+ qb t)) (aref K (+ kb t))))))
                         (set acc (* acc scale))
                         (when (> acc mx) (set mx acc)))
                       ;; pass 2: the normaliser
                       (declare sm :float 0.0)
                       (for (j 0 n)
                         (declare kb :uint (+ (* j kvdim) kc))
                         (declare acc :float 0.0)
                         (for (t 0 hd)
                           (set acc (+ acc (* (aref Q (+ qb t)) (aref K (+ kb t))))))
                         (set sm (+ sm (exp (- (* acc scale) mx)))))
                       ;; pass 3: the context, accumulated in place
                       (for (t 0 hd) (store (aref CTX (+ qb t)) 0.0))
                       (for (j 0 n)
                         (declare kb :uint (+ (* j kvdim) kc))
                         (declare acc :float 0.0)
                         (for (t 0 hd)
                           (set acc (+ acc (* (aref Q (+ qb t)) (aref K (+ kb t))))))
                         (declare pj :float (/ (exp (- (* acc scale) mx)) sm))
                         (for (t 0 hd)
                           (store (aref CTX (+ qb t))
                                  (+ (aref CTX (+ qb t)) (* pj (aref V (+ kb t)))))))))))
      nelisp-gpu-kernels)

;; --- BitNet b1.58 Phase B: packed ternary-weight matmul ---------------------
;; Ternary weights are packed as base-4 codes (tern+1 in {0,1,2}), PK codes per
;; f32 (so the weight buffer is PK x smaller -- bandwidth/VRAM win on Pascal).
;; Y (seq x out) = X (seq x in, f32) . Wq^T + BIAS, Wq = BETA[0]*ternary, where the
;; weight row o is FCOUNT packed floats (FCOUNT = ceil(in/PK)); each packed float
;; is read ONCE and its PK codes peeled off with %4 // /4 (no bitcast needed).
(push (cons 'bitlinear-packed
            '(:buffers (X WPK BIAS BETA Y) :push (seq in out pk fcount) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq out))
                       (declare s :uint (/ idx out)) (declare o :uint (% idx out))
                       (declare acc :float (aref BIAS o)) (declare beta :float (aref BETA 0))
                       (declare xb :uint (* s in)) (declare rowf :uint (* o fcount))
                       (for (f 0 fcount)
                         (declare packed :uint (uint (aref WPK (+ rowf f))))
                         (declare i0 :uint (* f pk))
                         (for (z 0 pk)
                           (declare i :uint (+ i0 z))
                           (when (< i in)
                             (declare code :uint (% packed 4))
                             (declare tern :float (- (float code) 1.0))
                             (set acc (+ acc (* (aref X (+ xb i)) (* beta tern)))))
                           (set packed (/ packed 4))))
                       (store (aref Y idx) acc)))))
      nelisp-gpu-kernels)

;; Per-row-beta packed ternary linear: like bitlinear-packed but BETA is indexed
;; by the output row, so several projections with DIFFERENT absmean scales (e.g.
;; Q|K|V or gate|up) can be concatenated into ONE packed weight and run in ONE
;; dispatch (fused multi-projection -> fewer dispatches per token).
(push (cons 'bitlinear-packed-v
            '(:buffers (X WPK BIAS BETA Y) :push (seq in out pk fcount) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq out))
                       (declare s :uint (/ idx out)) (declare o :uint (% idx out))
                       (declare acc :float (aref BIAS o)) (declare beta :float (aref BETA o))
                       (declare xb :uint (* s in)) (declare rowf :uint (* o fcount))
                       (for (f 0 fcount)
                         (declare packed :uint (uint (aref WPK (+ rowf f))))
                         (declare i0 :uint (* f pk))
                         (for (z 0 pk)
                           (declare i :uint (+ i0 z))
                           (when (< i in)
                             (declare code :uint (% packed 4))
                             (declare tern :float (- (float code) 1.0))
                             (set acc (+ acc (* (aref X (+ xb i)) (* beta tern)))))
                           (set packed (/ packed 4))))
                       (store (aref Y idx) acc)))))
      nelisp-gpu-kernels)

;; Append one token's raw K/V (taken from a fused QKV buffer at offsets dim and
;; dim+kvdim) into the resident paged streaming pools CK/CV at physical base WBASE.
(push (cons 'qkv-append
            '(:buffers (QKV CK CV) :push (dim kvdim wbase) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx kvdim)
                       (store (aref CK (+ wbase idx)) (aref QKV (+ dim idx)))
                       (store (aref CV (+ wbase idx)) (aref QKV (+ dim (+ kvdim idx))))))))
      nelisp-gpu-kernels)

;; GPU streaming attention driven by a precomputed entry list ENT (ne pairs of
;; [physical-base, cache-relative-pos]): one thread per output dim attends every
;; kept sink+window entry, RoPE'ing K by its cache-relative pos and Q by QCREL in
;; the kernel (CO/SI = cos/sin tables indexed [crel*half + m], pairs (2m,2m+1) to
;; match nl-llm--rope-block).  Two-pass softmax, no per-thread arrays.  GQA aware.
(push (cons 'attn-stream-entries
            '(:buffers (Q CK CV ENT CO SI C) :push (dim heads kvheads ne qcrel) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare half :uint (/ hd 2))
                     (when (< idx dim)
                       (declare h :uint (/ idx hd)) (declare t :uint (% idx hd)) (declare kvh :uint (/ h grp))
                       (declare c0q :uint (* h hd)) (declare coef :float (/ 1.0 (sqrt (float hd))))
                       (declare mx :float -1.0e30)
                       (for (r 0 ne)
                         (declare base :uint (uint (aref ENT (* 2 r)))) (declare crel :uint (uint (aref ENT (+ (* 2 r) 1))))
                         (declare kb :uint (+ base (* kvh hd))) (declare acc :float 0.0)
                         (for (m 0 half)
                           (declare ci :uint (+ (* crel half) m)) (declare cc :float (aref CO ci)) (declare ss :float (aref SI ci))
                           (declare k0 :float (aref CK (+ kb (* 2 m)))) (declare k1 :float (aref CK (+ kb (+ (* 2 m) 1))))
                           (declare kr0 :float (- (* k0 cc) (* k1 ss))) (declare kr1 :float (+ (* k0 ss) (* k1 cc)))
                           (declare qci :uint (+ (* qcrel half) m)) (declare qcc :float (aref CO qci)) (declare qss :float (aref SI qci))
                           (declare q0 :float (aref Q (+ c0q (* 2 m)))) (declare q1 :float (aref Q (+ c0q (+ (* 2 m) 1))))
                           (declare qr0 :float (- (* q0 qcc) (* q1 qss))) (declare qr1 :float (+ (* q0 qss) (* q1 qcc)))
                           (set acc (+ acc (+ (* qr0 kr0) (* qr1 kr1)))))
                         (declare s :float (* acc coef)) (when (> s mx) (set mx s)))
                       (declare z :float 0.0) (declare ctx :float 0.0)
                       (for (r 0 ne)
                         (declare base :uint (uint (aref ENT (* 2 r)))) (declare crel :uint (uint (aref ENT (+ (* 2 r) 1))))
                         (declare kb :uint (+ base (* kvh hd))) (declare acc :float 0.0)
                         (for (m 0 half)
                           (declare ci :uint (+ (* crel half) m)) (declare cc :float (aref CO ci)) (declare ss :float (aref SI ci))
                           (declare k0 :float (aref CK (+ kb (* 2 m)))) (declare k1 :float (aref CK (+ kb (+ (* 2 m) 1))))
                           (declare kr0 :float (- (* k0 cc) (* k1 ss))) (declare kr1 :float (+ (* k0 ss) (* k1 cc)))
                           (declare qci :uint (+ (* qcrel half) m)) (declare qcc :float (aref CO qci)) (declare qss :float (aref SI qci))
                           (declare q0 :float (aref Q (+ c0q (* 2 m)))) (declare q1 :float (aref Q (+ c0q (+ (* 2 m) 1))))
                           (declare qr0 :float (- (* q0 qcc) (* q1 qss))) (declare qr1 :float (+ (* q0 qss) (* q1 qcc)))
                           (set acc (+ acc (+ (* qr0 kr0) (* qr1 kr1)))))
                         (declare e :float (exp (- (* acc coef) mx))) (set z (+ z e))
                         (set ctx (+ ctx (* e (aref CV (+ base (+ (* kvh hd) t)))))))
                       (store (aref C idx) (/ ctx z))))))
      nelisp-gpu-kernels)

;; --- PagedAttention: block-paged KV cache (decode) --------------------------
;; KV lives in a shared POOL of fixed-size blocks; a per-sequence block TABLE
;; (B x mbps) maps logical block -> physical block id.  The physical slot for
;; sequence s position pos is block TABLE[s*mbps + pos/bs], offset pos%bs.
;; Append B rows SRC (B x kvdim) into the POOL through the TABLE at POS[0].
(push (cons 'cache-append-paged
            '(:buffers (SRC POS TABLE POOL) :push (B kvdim bs mbps) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* B kvdim))
                       (declare s :uint (/ idx kvdim)) (declare t :uint (% idx kvdim))
                       (declare pos :uint (uint (aref POS 0)))
                       (declare lb :uint (/ pos bs)) (declare off :uint (% pos bs))
                       (declare phys :uint (uint (aref TABLE (+ (* s mbps) lb))))
                       (store (aref POOL (+ (* (+ (* phys bs) off) kvdim) t)) (aref SRC idx))))))
      nelisp-gpu-kernels)
;; Per-sequence single-query attention over the paged POOL: C (B x dim).
;; Sequence s attends positions 0..POS[0], gathering K/V through the block TABLE.
(push (cons 'decode-attn-paged
            '(:buffers (Q CK CV POS TABLE C) :push (B dim heads kvheads bs mbps) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* B dim))
                       (declare s :uint (/ idx dim)) (declare c :uint (% idx dim))
                       (declare h :uint (/ c hd)) (declare t :uint (% c hd)) (declare kvh :uint (/ h grp))
                       (declare pos :uint (uint (aref POS 0))) (declare len :uint (+ pos 1))
                       (declare c0q :uint (+ (* s dim) (* h hd))) (declare coef :float (/ 1.0 (sqrt (float hd))))
                       (declare mx :float -1.0e30)
                       (for (j 0 len)
                         (declare lb :uint (/ j bs)) (declare off :uint (% j bs))
                         (declare phys :uint (uint (aref TABLE (+ (* s mbps) lb))))
                         (declare kb :uint (+ (* (+ (* phys bs) off) kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare sc :float (* acc coef)) (when (> sc mx) (set mx sc)))
                       (declare z :float 0.0) (declare ctx :float 0.0)
                       (for (j 0 len)
                         (declare lb :uint (/ j bs)) (declare off :uint (% j bs))
                         (declare phys :uint (uint (aref TABLE (+ (* s mbps) lb))))
                         (declare kb :uint (+ (* (+ (* phys bs) off) kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare e :float (exp (- (* acc coef) mx))) (set z (+ z e))
                         (set ctx (+ ctx (* e (aref CV (+ kb t))))))
                       (store (aref C idx) (/ ctx z))))))
      nelisp-gpu-kernels)

;; --- Tree-attention: batched draft-tree verify in one dispatch --------------
;; M draft-tree nodes each attend over a SHARED context cache CK/CV (LEN[0]
;; entries) plus their ancestor chain among the nodes (NK/NV, walked via the
;; parent array PAR, sentinel = M for a root).  Q/CK/NK are pre-RoPE'd; this is
;; pure masked attention -- the core of a one-forward tree verify.  GQA: kv head
;; = h/grp.  MAXDEPTH bounds the chain walk.
(push (cons 'tree-attn
            '(:buffers (Q CK CV NK NV PAR LEN C) :push (M dim heads kvheads maxdepth) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* M dim))
                       (declare i :uint (/ idx dim)) (declare c :uint (% idx dim))
                       (declare h :uint (/ c hd)) (declare t :uint (% c hd)) (declare kvh :uint (/ h grp))
                       (declare L :uint (uint (aref LEN 0)))
                       (declare c0q :uint (+ (* i dim) (* h hd))) (declare coef :float (/ 1.0 (sqrt (float hd))))
                       (declare mx :float -1.0e30)
                       (for (j 0 L)
                         (declare kb :uint (+ (* j kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare s :float (* acc coef)) (when (> s mx) (set mx s)))
                       (declare cur :uint i)
                       (for (d 0 maxdepth)
                         (when (< cur M)
                           (declare kb :uint (+ (* cur kvdim) (* kvh hd))) (declare acc :float 0.0)
                           (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref NK (+ kb tt))))))
                           (declare s :float (* acc coef)) (when (> s mx) (set mx s))
                           (set cur (uint (aref PAR cur)))))
                       (declare z :float 0.0) (declare ctx :float 0.0)
                       (for (j 0 L)
                         (declare kb :uint (+ (* j kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare e :float (exp (- (* acc coef) mx))) (set z (+ z e))
                         (set ctx (+ ctx (* e (aref CV (+ kb t))))))
                       (declare cur2 :uint i)
                       (for (d 0 maxdepth)
                         (when (< cur2 M)
                           (declare kb :uint (+ (* cur2 kvdim) (* kvh hd))) (declare acc :float 0.0)
                           (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref NK (+ kb tt))))))
                           (declare e :float (exp (- (* acc coef) mx))) (set z (+ z e))
                           (set ctx (+ ctx (* e (aref NV (+ kb t)))))
                           (set cur2 (uint (aref PAR cur2)))))
                       (store (aref C idx) (/ ctx z))))))
      nelisp-gpu-kernels)

;; Copy one block (bs*kvdim floats) from physical block SRC to DST within a paged
;; POOL, in place -- the GPU side of copy-on-write for a shared partial prefix block.
(push (cons 'block-copy
            '(:buffers (POOL) :push (bs kvdim src dst) :local-size 64
              :body ((declare i :uint (gid-x)) (declare bn :uint (* bs kvdim))
                     (when (< i bn)
                       (store (aref POOL (+ (* dst bn) i)) (aref POOL (+ (* src bn) i)))))))
      nelisp-gpu-kernels)

;; --- PagedAttention variable-length: per-sequence positions (LENS[s]) --------
;; Like the paged kernels but each sequence b decodes at its OWN length LENS[b]
;; (not a shared POS[0]), so a batch can hold sequences of different lengths and
;; share prefix blocks.  RoPE row b at LENS[b].
(push (cons 'decode-rope-b-v
            '(:buffers (X CO SI SG LENS C) :push (B cols heads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ cols heads)) (declare half :uint (/ hd 2)) (declare rh :uint (* heads half))
                     (when (< idx (* B rh))
                       (declare bb :uint (/ idx rh)) (declare rem :uint (% idx rh))
                       (declare m :uint (% rem half)) (declare h :uint (/ rem half))
                       (declare pos :uint (uint (aref LENS bb))) (declare ci :uint (+ (* pos half) m))
                       (declare cc :float (aref CO ci)) (declare ss :float (* (aref SI ci) (aref SG 0)))
                       (declare b0 :uint (+ (* bb cols) (* h hd))) (declare i0 :uint (+ b0 (* 2 m))) (declare i1 :uint (+ i0 1))
                       (declare a0 :float (aref X i0)) (declare a1 :float (aref X i1))
                       (store (aref C i0) (- (* a0 cc) (* a1 ss)))
                       (store (aref C i1) (+ (* a0 ss) (* a1 cc)))))))
      nelisp-gpu-kernels)
;; Append B rows SRC into the paged POOL, each at its sequence's length LENS[b].
(push (cons 'cache-append-paged-v
            '(:buffers (SRC LENS TABLE POOL) :push (B kvdim bs mbps) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* B kvdim))
                       (declare s :uint (/ idx kvdim)) (declare t :uint (% idx kvdim))
                       (declare pos :uint (uint (aref LENS s)))
                       (declare lb :uint (/ pos bs)) (declare off :uint (% pos bs))
                       (declare phys :uint (uint (aref TABLE (+ (* s mbps) lb))))
                       (store (aref POOL (+ (* (+ (* phys bs) off) kvdim) t)) (aref SRC idx))))))
      nelisp-gpu-kernels)
;; Per-sequence paged attention: sequence s attends 0..LENS[s] through its table.
(push (cons 'decode-attn-paged-v
            '(:buffers (Q CK CV LENS TABLE C) :push (B dim heads kvheads bs mbps) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare grp :uint (/ heads kvheads)) (declare kvdim :uint (* kvheads hd))
                     (when (< idx (* B dim))
                       (declare s :uint (/ idx dim)) (declare c :uint (% idx dim))
                       (declare h :uint (/ c hd)) (declare t :uint (% c hd)) (declare kvh :uint (/ h grp))
                       (declare pos :uint (uint (aref LENS s))) (declare len :uint (+ pos 1))
                       (declare c0q :uint (+ (* s dim) (* h hd))) (declare coef :float (/ 1.0 (sqrt (float hd))))
                       (declare mx :float -1.0e30)
                       (for (j 0 len)
                         (declare lb :uint (/ j bs)) (declare off :uint (% j bs))
                         (declare phys :uint (uint (aref TABLE (+ (* s mbps) lb))))
                         (declare kb :uint (+ (* (+ (* phys bs) off) kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare sc :float (* acc coef)) (when (> sc mx) (set mx sc)))
                       (declare z :float 0.0) (declare ctx :float 0.0)
                       (for (j 0 len)
                         (declare lb :uint (/ j bs)) (declare off :uint (% j bs))
                         (declare phys :uint (uint (aref TABLE (+ (* s mbps) lb))))
                         (declare kb :uint (+ (* (+ (* phys bs) off) kvdim) (* kvh hd))) (declare acc :float 0.0)
                         (for (tt 0 hd) (set acc (+ acc (* (aref Q (+ c0q tt)) (aref CK (+ kb tt))))))
                         (declare e :float (exp (- (* acc coef) mx))) (set z (+ z e))
                         (set ctx (+ ctx (* e (aref CV (+ kb t))))))
                       (store (aref C idx) (/ ctx z))))))
      nelisp-gpu-kernels)

;; --- modern-block training kernels (RMSNorm / RoPE / SwiGLU / CE) -----
;; Per-row inverse RMS: S[r] = 1/sqrt(mean(X[r]^2) + 1e-6).
(push (cons 'rmsnorm-istd
            '(:buffers (X S) :push (M N) :local-size 64
              :body ((declare r :uint (gid-x))
                     (when (< r M)
                       (declare base :uint (* r N)) (declare ss :float 0.0)
                       (for (k 0 N) (declare v :float (aref X (+ base k))) (set ss (+ ss (* v v))))
                       (store (aref S r) (/ 1.0 (sqrt (+ (/ ss (float N)) 0.000001))))))))
      nelisp-gpu-kernels)
;; RMSNorm forward: C[r,j] = X[r,j] * S[r] * G[j].
(push (cons 'rmsnorm-fwd
            '(:buffers (X S G C) :push (M N) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* M N))
                       (declare r :uint (/ idx N)) (declare j :uint (% idx N))
                       (store (aref C idx) (* (* (aref X idx) (aref S r)) (aref G j)))))))
      nelisp-gpu-kernels)
;; RMSNorm dx: per row, a=sum_k DY*G*X; DX[r,j]=S*DY*G - S^3*X*a/N.
(push (cons 'rmsnorm-dx
            '(:buffers (DY X S G DX) :push (M N) :local-size 64
              :body ((declare r :uint (gid-x))
                     (when (< r M)
                       (declare base :uint (* r N)) (declare is :float (aref S r)) (declare a :float 0.0)
                       (for (k 0 N) (set a (+ a (* (* (aref DY (+ base k)) (aref G k)) (aref X (+ base k))))))
                       (declare is3 :float (/ (* (* is is) is) (float N)))
                       (for (j 0 N)
                         (store (aref DX (+ base j))
                                (- (* (* is (aref DY (+ base j))) (aref G j))
                                   (* (* is3 (aref X (+ base j))) a))))))))
      nelisp-gpu-kernels)
;; RMSNorm dgamma: DG[j] = sum_r DY[r,j]*X[r,j]*S[r].
(push (cons 'rmsnorm-dgamma
            '(:buffers (DY X S DG) :push (M N) :local-size 64
              :body ((declare j :uint (gid-x))
                     (when (< j N)
                       (declare acc :float 0.0)
                       (for (r 0 M) (set acc (+ acc (* (* (aref DY (+ (* r N) j)) (aref X (+ (* r N) j))) (aref S r)))))
                       (store (aref DG j) acc)))))
      nelisp-gpu-kernels)
;; RoPE apply with host-precomputed cos/sin tables CO,SI (seq x half), sign SG[0]
;; (+1 forward, -1 backward = inverse rotation).  Rotates each (i0,i1) pair.
(push (cons 'rope-apply
            '(:buffers (X CO SI SG C) :push (seq dim heads) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (declare hd :uint (/ dim heads)) (declare half :uint (/ hd 2))
                     (when (< idx (* (* seq heads) half))
                       (declare m :uint (% idx half)) (declare tmp :uint (/ idx half))
                       (declare h :uint (% tmp heads)) (declare p :uint (/ tmp heads))
                       (declare ci :uint (+ (* p half) m))
                       (declare cc :float (aref CO ci)) (declare ss :float (* (aref SI ci) (aref SG 0)))
                       (declare b0 :uint (+ (* p dim) (* h hd)))
                       (declare i0 :uint (+ b0 (* 2 m))) (declare i1 :uint (+ i0 1))
                       (declare a0 :float (aref X i0)) (declare a1 :float (aref X i1))
                       (store (aref C i0) (- (* a0 cc) (* a1 ss)))
                       (store (aref C i1) (+ (* a0 ss) (* a1 cc)))))))
      nelisp-gpu-kernels)
;; SiLU forward: C[i] = X[i] * sigmoid(X[i]).
(push (cons 'silu
            '(:buffers (X C) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (declare x :float (aref X i)) (declare sg :float (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))
                       (store (aref C i) (* x sg))))))
      nelisp-gpu-kernels)
;; SiLU backward: D[i] = G[i] * (sg*(1 + x*(1-sg))).
(push (cons 'silu-bwd
            '(:buffers (G X D) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (declare x :float (aref X i)) (declare sg :float (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))
                       (store (aref D i) (* (aref G i) (* sg (+ 1.0 (* x (- 1.0 sg))))))))))
      nelisp-gpu-kernels)
;; Elementwise product: C[i] = A[i] * B[i].
(push (cons 'mul
            '(:buffers (A B C) :push (n) :local-size 64
              :body ((declare i :uint (gid-x)) (when (< i n) (store (aref C i) (* (aref A i) (aref B i)))))))
      nelisp-gpu-kernels)
;; Row-softmax backward: DS[r,j] = P[r,j]*(G[r,j] - sum_k P[r,k]*G[r,k]).
(push (cons 'softmax-bwd
            '(:buffers (P G DS) :push (M N) :local-size 64
              :body ((declare r :uint (gid-x))
                     (when (< r M)
                       (declare base :uint (* r N)) (declare dot :float 0.0)
                       (for (k 0 N) (set dot (+ dot (* (aref P (+ base k)) (aref G (+ base k))))))
                       (for (j 0 N) (store (aref DS (+ base j)) (* (aref P (+ base j)) (- (aref G (+ base j)) dot))))))))
      nelisp-gpu-kernels)
;; In-place Adam: m,v resident (persist + update each step); H = [lr_t, b1, b2,
;; eps] resident (lr_t = lr*sqrt(1-b2^t)/(1-b1^t) refreshed per step on the host).
;;   m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2 ; w -= lr_t * m/(sqrt(v)+eps).
(push (cons 'adam
            '(:buffers (W G M V H S) :push (n) :local-size 64
              :body ((declare i :uint (gid-x))
                     (when (< i n)
                       (declare g :float (* (aref S 0) (aref G i)))
                       (declare b1 :float (aref H 1)) (declare b2 :float (aref H 2))
                       (declare m :float (+ (* b1 (aref M i)) (* (- 1.0 b1) g)))
                       (declare v :float (+ (* b2 (aref V i)) (* (- 1.0 b2) (* g g))))
                       (store (aref M i) m) (store (aref V i) v)
                       (store (aref W i) (- (aref W i) (* (aref H 0) (/ m (+ (sqrt v) (aref H 3))))))))))
      nelisp-gpu-kernels)

;; --- MoE (top-k routing) kernels -------------------------------------
;; Top-K selection mask: MK[r,e] = 0 if logit LG[r,e] is among the row's K
;; largest, else -1e30 (added to the router logits before softmax).
(push (cons 'topk-mask
            '(:buffers (LG MK) :push (M E K) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* M E))
                       (declare r :uint (/ idx E)) (declare e :uint (% idx E))
                       (declare base :uint (* r E)) (declare ve :float (aref LG (+ base e)))
                       (declare cnt :uint 0)
                       (for (j 0 E) (when (> (aref LG (+ base j)) ve) (set cnt (+ cnt 1))))
                       (when (< cnt K) (store (aref MK idx) 0.0))
                       (when (< K (+ cnt 1)) (store (aref MK idx) -1.0e30))))))
      nelisp-gpu-kernels)
;; Scale rows: C[r,j] = Y[r,j] * S[r]  (per-row scalar gate).
(push (cons 'scale-rows
            '(:buffers (Y S C) :push (M N) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* M N))
                       (declare r :uint (/ idx N))
                       (store (aref C idx) (* (aref Y idx) (aref S r)))))))
      nelisp-gpu-kernels)
;; Row dot: D[r] = sum_j G[r,j] * Y[r,j]  (gradient of the per-row gate scalar).
(push (cons 'rowdot
            '(:buffers (G Y D) :push (M N) :local-size 64
              :body ((declare r :uint (gid-x))
                     (when (< r M)
                       (declare base :uint (* r N)) (declare acc :float 0.0)
                       (for (j 0 N) (set acc (+ acc (* (aref G (+ base j)) (aref Y (+ base j))))))
                       (store (aref D r) acc)))))
      nelisp-gpu-kernels)

;; --- gather embedding (token indices instead of one-hot) -------------
;; Forward gather: C[i,j] = WTE[TOK[i], j].  TOK is a (seq) float buffer of
;; token indices (exact for indices < 2^24).
(push (cons 'embed-gather
            '(:buffers (TOK WTE C) :push (seq dim) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* seq dim))
                       (declare i :uint (/ idx dim)) (declare j :uint (% idx dim))
                       (declare tk :uint (uint (aref TOK i)))
                       (store (aref C idx) (aref WTE (+ (* tk dim) j)))))))
      nelisp-gpu-kernels)
;; Backward scatter-add: DW[t,j] = sum_{i: TOK[i]=t} GX[i,j].  One thread per
;; (t,j); no atomics needed (each output element owned by one thread).
(push (cons 'embed-bwd
            '(:buffers (TOK GX DW) :push (seq dim vocab) :local-size 64
              :body ((declare idx :uint (gid-x))
                     (when (< idx (* vocab dim))
                       (declare tt :uint (/ idx dim)) (declare j :uint (% idx dim))
                       (declare acc :float 0.0)
                       (for (i 0 seq)
                         (when (= (uint (aref TOK i)) tt)
                           (set acc (+ acc (aref GX (+ (* i dim) j))))))
                       (store (aref DW idx) acc)))))
      nelisp-gpu-kernels)
;; CE gradient from a target-index buffer TG (M): G[r,j]=(softmax-[[j=tg]])/M.
(push (cons 'ce-grad-idx
            '(:buffers (LG TG G) :push (M V) :local-size 64
              :body ((declare r :uint (gid-x))
                     (when (< r M)
                       (declare base :uint (* r V)) (declare mx :float (aref LG base))
                       (for (j 0 V) (declare v :float (aref LG (+ base j))) (when (> v mx) (set mx v)))
                       (declare s :float 0.0)
                       (for (j2 0 V) (set s (+ s (exp (- (aref LG (+ base j2)) mx)))))
                       (declare tgt :uint (uint (aref TG r))) (declare invm :float (/ 1.0 (float M)))
                       (for (j3 0 V)
                         (declare sub :float 0.0)
                         (when (= j3 tgt) (set sub 1.0))
                         (store (aref G (+ base j3))
                                (* (- (/ (exp (- (aref LG (+ base j3)) mx)) s) sub) invm)))))))
      nelisp-gpu-kernels)

;; Softmax cross-entropy gradient: G[r,j] = (softmax(LG)[r,j] - OH[r,j]) / M
;; where OH is the one-hot target matrix (M x V).
(push (cons 'ce-grad
            '(:buffers (LG OH G) :push (M V) :local-size 64
              :body ((declare r :uint (gid-x))
                     (when (< r M)
                       (declare base :uint (* r V)) (declare mx :float (aref LG base))
                       (for (j 0 V) (declare v :float (aref LG (+ base j))) (when (> v mx) (set mx v)))
                       (declare s :float 0.0)
                       (for (j2 0 V) (set s (+ s (exp (- (aref LG (+ base j2)) mx)))))
                       (declare invm :float (/ 1.0 (float M)))
                       (for (j3 0 V)
                         (store (aref G (+ base j3))
                                (* (- (/ (exp (- (aref LG (+ base j3)) mx)) s) (aref OH (+ base j3))) invm)))))))
      nelisp-gpu-kernels)
