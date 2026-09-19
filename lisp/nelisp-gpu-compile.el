;;; nelisp-gpu-compile.el --- s-expr kernel DSL -> SPIR-V compiler  -*- lexical-binding: t; -*-

;; Phase 2 of nelisp-gpu: a small compiler that lowers a compute kernel
;; written as elisp s-expressions to a SPIR-V binary module, which runs
;; as native GPU code under Vulkan (host/vkcompute, host/vkmatmul).
;;
;; Kernel spec (plist):
;;   :buffers  (A B C ...)   float storage buffers, set 0 bindings 0..k
;;   :push     (n) | (M K N) uint push constants (members 0..)
;;   :local-size 64          workgroup size along x
;;   :body     (stmt ...)
;;
;; Statements:
;;   (declare NAME :uint|:float EXPR)   function-local var + init store
;;   (set NAME EXPR)                    store to a local
;;   (store (aref BUF EXPR) EXPR)       store to a buffer element
;;   (when COND STMT...)               structured selection
;;   (for (VAR LO HI) STMT...)         structured loop, VAR a local uint
;; Expressions:
;;   <int>          uint constant      <float>  float constant
;;   (gid-x)        gl_GlobalInvocationID.x  (uint)
;;   NAME           local var / push-constant ref
;;   (aref BUF E)   buffer load (float)
;;   (+ - * / %)    arithmetic (int or float by operand type)
;;   (< <= > >= =)  unsigned/int comparison -> bool

;;; Code:

(require 'cl-lib)

;; --- compilation state (dynamically bound in `nelisp-gpu-compile') ----
(defvar nelisp-gpu--next)      ; next result id
(defvar nelisp-gpu--decos)     ; decoration instructions (word lists)
(defvar nelisp-gpu--globals)   ; types/consts/global-var instructions
(defvar nelisp-gpu--cache)     ; equal-hash: type/const key -> id
(defvar nelisp-gpu--locals)    ; OpVariable (Function) instructions, emitted at fn top
(defvar nelisp-gpu--body)      ; function body instructions
(defvar nelisp-gpu--env)       ; alist NAME -> (varid . type-kw)   (locals)
(defvar nelisp-gpu--push)      ; alist NAME -> member-index         (push consts)
(defvar nelisp-gpu--gid)       ; id of gl_GlobalInvocationID variable
(defvar nelisp-gpu--main)      ; id of the entry function
(defvar nelisp-gpu--bufs)      ; alist NAME -> buffer variable id
(defvar nelisp-gpu--pcvar)     ; push-constant variable id
(defvar nelisp-gpu--glsl)      ; id of the GLSL.std.450 ext instruction set

(defun nelisp-gpu--newid () (prog1 nelisp-gpu--next (cl-incf nelisp-gpu--next)))

(defun nelisp-gpu--w (op operands)
  "Word list for one SPIR-V instruction OP with OPERANDS."
  (cons (logior (ash (1+ (length operands)) 16) op) operands))

(defun nelisp-gpu--global (op operands)
  (push (nelisp-gpu--w op operands) nelisp-gpu--globals))
(defun nelisp-gpu--deco (op operands)
  (push (nelisp-gpu--w op operands) nelisp-gpu--decos))
(defun nelisp-gpu--emit (op operands)
  (push (nelisp-gpu--w op operands) nelisp-gpu--body))

(defun nelisp-gpu--string-words (str)
  (let ((bytes (append (string-to-list str) (list 0))) words)
    (while (/= (% (length bytes) 4) 0) (setq bytes (append bytes (list 0))))
    (let ((b bytes))
      (while b
        (push (logior (nth 0 b) (ash (nth 1 b) 8)
                      (ash (nth 2 b) 16) (ash (nth 3 b) 24)) words)
        (setq b (nthcdr 4 b))))
    (nreverse words)))

;; --- type / constant interning ---------------------------------------
(defun nelisp-gpu--type (key)
  "Return the SPIR-V id for type KEY, building (and caching) it once.
Dependencies are built first so they precede KEY in the binary stream."
  (or (gethash key nelisp-gpu--cache)
      (let ((id (nelisp-gpu--newid)))
        (pcase key
          (:void   (nelisp-gpu--global 19 (list id)))
          (:bool   (nelisp-gpu--global 20 (list id)))
          (:uint   (nelisp-gpu--global 21 (list id 32 0)))
          (:int    (nelisp-gpu--global 21 (list id 32 1)))   ; signed 32-bit (for OpSDot)
          (:float  (nelisp-gpu--global 22 (list id 32)))
          (:v3uint (nelisp-gpu--global 23 (list id (nelisp-gpu--type :uint) 3)))
          (:fnvoid (nelisp-gpu--global 33 (list id (nelisp-gpu--type :void))))
          (:rtarr  (nelisp-gpu--global 29 (list id (nelisp-gpu--type :float)))
                   (nelisp-gpu--deco 71 (list id 6 4)))          ; ArrayStride 4
          (:ssbo   (let ((arr (nelisp-gpu--type :rtarr)))
                     (nelisp-gpu--global 30 (list id arr))
                     (nelisp-gpu--deco 71 (list id 3))           ; BufferBlock
                     (nelisp-gpu--deco 72 (list id 0 35 0))))    ; member0 Offset 0
          (`(:ptr ,sc ,ty) (nelisp-gpu--global 32 (list id sc (nelisp-gpu--type ty))))
          (`(:ptrid ,sc ,tyid) (nelisp-gpu--global 32 (list id sc tyid)))
          (`(:pcstruct ,n)
           (let ((u (nelisp-gpu--type :uint)) members)
             (dotimes (_ n) (push u members))
             (nelisp-gpu--global 30 (cons id (nreverse members)))
             (nelisp-gpu--deco 71 (list id 2))                   ; Block
             (dotimes (i n) (nelisp-gpu--deco 72 (list id i 35 (* 4 i))))))
          (`(:arrf ,size)                       ; fixed float array (workgroup)
           (nelisp-gpu--global 28 (list id (nelisp-gpu--type :float)
                                        (nelisp-gpu--const-uint size))))
          (`(:ptr-wg-arr ,size)                 ; pointer to workgroup float array
           (nelisp-gpu--global 32 (list id 4 (nelisp-gpu--type (list :arrf size)))))
          (_ (error "nelisp-gpu: unknown type key %S" key)))
        (puthash key id nelisp-gpu--cache)
        id)))

(defun nelisp-gpu--const-uint (v)
  (let ((key (list :cu v)))
    (or (gethash key nelisp-gpu--cache)
        (let ((id (nelisp-gpu--newid)))
          (nelisp-gpu--global 43 (list (nelisp-gpu--type :uint) id v))
          (puthash key id nelisp-gpu--cache) id))))

(defconst nelisp-gpu--f32-min-normal 1.1754943508222875e-38
  "Smallest positive normal float32, 2^-126.")
(defconst nelisp-gpu--f32-min-subnormal 1.401298464324817e-45
  "Smallest positive subnormal float32, 2^-149; also the subnormal step.")
(defconst nelisp-gpu--f32-overflow 3.4028235677973366e38
  "Smallest double that rounds to float32 infinity, (2 - 2^-24) * 2^127.")

(defun nelisp-gpu--f32-bits (x)
  "IEEE-754 single-precision bit pattern of float X as a uint.

Magnitudes outside float32's range are handled explicitly, and that is the
whole point of the function rather than a detail.  An earlier version masked
the exponent with #xff and nothing else, so a number too small to represent
came back *enormous*: 3.7e-44 encoded to 4.3e+33, 1e-39 to 1.2e+38, and a
number too large came back tiny, 1e+300 to 5.6e-09.  Nothing signalled.

That is reachable from ordinary data, not just extremes.  A softmax over a
large vocabulary puts most of its mass below 1e-38, so a gradient built from
one arrived at the GPU as garbage of order 1e33 -- which is how it was found:
a transpose that agreed with the CPU to 1e-05 on pseudo-random input
disagreed by 7.7e+30 on a real one."
  (setq x (float x))
  (cond
   ((isnan x) #x7fc00000)
   ((= x 0.0) 0)
   (t
    (let ((sign (if (< x 0.0) 1 0)) (a (abs x)))
      (cond
       ;; Too large: an infinity, which at least propagates visibly.
       ((>= a nelisp-gpu--f32-overflow) (logior (ash sign 31) #x7f800000))
       ;; Too small to be normal: a subnormal, or zero if it rounds there.
       ((< a nelisp-gpu--f32-min-normal)
        (let ((m (round (/ a nelisp-gpu--f32-min-subnormal))))
          (if (>= m 8388608)                  ; rounded up into the normals
              (logior (ash sign 31) (ash 1 23))
            (logior (ash sign 31) m))))
       (t
        (let* ((fe (frexp a)) (s (car fe)) (e (cdr fe))
               (m2 (* 2.0 s)) (ee (+ e 126))
               (m (round (* (- m2 1.0) 8388608.0))))
          (when (>= m 8388608) (setq m 0 ee (1+ ee)))
          (if (>= ee 255)
              (logior (ash sign 31) #x7f800000)
            (logior (ash sign 31) (ash ee 23) (logand m #x7fffff))))))))))

(defun nelisp-gpu--const-float (v)
  (let ((key (list :cf v)))
    (or (gethash key nelisp-gpu--cache)
        (let ((id (nelisp-gpu--newid)))
          (nelisp-gpu--global 43 (list (nelisp-gpu--type :float) id
                                       (nelisp-gpu--f32-bits v)))
          (puthash key id nelisp-gpu--cache) id))))

;; --- expression lowering: returns (ID . TYPE-KW) ----------------------
(defun nelisp-gpu--ptr-uniform-float () (nelisp-gpu--type '(:ptr 2 :float)))

(defun nelisp-gpu--buffer-elt-ptr (bufvar idx-id)
  "AccessChain to BUFVAR's float element at IDX-ID -> pointer id."
  (let ((pid (nelisp-gpu--newid)))
    (nelisp-gpu--emit 65 (list (nelisp-gpu--ptr-uniform-float) pid
                               bufvar (nelisp-gpu--const-uint 0) idx-id))
    pid))

(defun nelisp-gpu--arith-op (sym type)
  (pcase (cons sym type)
    (`(+ . :uint) 128) (`(+ . :float) 129) (`(+ . :int) 128)
    (`(- . :uint) 130) (`(- . :float) 131) (`(- . :int) 130)
    (`(* . :uint) 132) (`(* . :float) 133) (`(* . :int) 132)
    (`(/ . :uint) 134) (`(/ . :float) 136)
    (`(% . :uint) 137)
    (_ (error "nelisp-gpu: bad arith %S on %S" sym type))))

(defun nelisp-gpu--cmp-op (sym float)
  "SPIR-V opcode for comparison SYM; FLOAT selects FOrd* over unsigned ops."
  (if float
      (pcase sym (`= 180) (`> 186) (`>= 190) (`< 184) (`<= 188)
             (_ (error "nelisp-gpu: bad cmp %S" sym)))
    (pcase sym (`= 170) (`> 172) (`>= 174) (`< 176) (`<= 178)
           (_ (error "nelisp-gpu: bad cmp %S" sym)))))

(defvar nelisp-gpu--lid)
(defvar nelisp-gpu--wgid)
(defvar nelisp-gpu--shared)
(defun nelisp-gpu--builtin-comp (var comp)
  "Load component COMP of v3uint builtin VAR -> (id . :uint)."
  (let ((v (nelisp-gpu--newid)) (id (nelisp-gpu--newid)))
    (nelisp-gpu--emit 61 (list (nelisp-gpu--type :v3uint) v var))
    (nelisp-gpu--emit 81 (list (nelisp-gpu--type :uint) id v comp))
    (cons id :uint)))

(defun nelisp-gpu--lower (e)
  (cond
   ((integerp e) (cons (nelisp-gpu--const-uint e) :uint))
   ((floatp e)   (cons (nelisp-gpu--const-float e) :float))
   ((symbolp e)
    (let ((loc (assq e nelisp-gpu--env)))
      (if loc
          (let* ((varid (cadr loc)) (ty (cddr loc)) (id (nelisp-gpu--newid)))
            (nelisp-gpu--emit 61 (list (nelisp-gpu--type ty) id varid))
            (cons id ty))
        (let ((mi (cdr (assq e nelisp-gpu--push))))
          (unless mi (error "nelisp-gpu: unbound %S" e))
          (let* ((pp (nelisp-gpu--newid)) (id (nelisp-gpu--newid))
                 (pcptr (nelisp-gpu--type '(:ptr 9 :uint))))
            (nelisp-gpu--emit 65 (list pcptr pp nelisp-gpu--pcvar
                                       (nelisp-gpu--const-uint mi)))
            (nelisp-gpu--emit 61 (list (nelisp-gpu--type :uint) id pp))
            (cons id :uint))))))
   ((eq (car-safe e) 'gid-x) (nelisp-gpu--builtin-comp nelisp-gpu--gid 0))
   ((eq (car-safe e) 'lid-x) (nelisp-gpu--builtin-comp nelisp-gpu--lid 0))
   ((eq (car-safe e) 'lid-y) (nelisp-gpu--builtin-comp nelisp-gpu--lid 1))
   ((eq (car-safe e) 'wgid-x) (nelisp-gpu--builtin-comp nelisp-gpu--wgid 0))
   ((eq (car-safe e) 'wgid-y) (nelisp-gpu--builtin-comp nelisp-gpu--wgid 1))
   ((eq (car-safe e) 'aref-shared)
    (let* ((sv (cdr (assq (nth 1 e) nelisp-gpu--shared)))
           (idx (car (nelisp-gpu--lower (nth 2 e))))
           (pid (nelisp-gpu--newid)) (id (nelisp-gpu--newid)))
      (unless sv (error "nelisp-gpu: unknown shared array %S" (nth 1 e)))
      (nelisp-gpu--emit 65 (list (nelisp-gpu--type '(:ptr 4 :float)) pid sv idx))
      (nelisp-gpu--emit 61 (list (nelisp-gpu--type :float) id pid))
      (cons id :float)))
   ((eq (car-safe e) 'aref)
    (let* ((bufvar (cdr (assq (nth 1 e) nelisp-gpu--bufs)))
           (idx (car (nelisp-gpu--lower (nth 2 e))))
           (pid (nelisp-gpu--buffer-elt-ptr bufvar idx))
           (id (nelisp-gpu--newid)))
      (unless bufvar (error "nelisp-gpu: unknown buffer %S" (nth 1 e)))
      (nelisp-gpu--emit 61 (list (nelisp-gpu--type :float) id pid))
      (cons id :float)))
   ((memq (car-safe e) '(+ - * / %))
    (let* ((a (nelisp-gpu--lower (nth 1 e))) (b (nelisp-gpu--lower (nth 2 e)))
           (ty (cdr a)) (id (nelisp-gpu--newid)))
      (unless (eq (cdr a) (cdr b))
        (error "nelisp-gpu: type mismatch in %S (%S vs %S)" e (cdr a) (cdr b)))
      (nelisp-gpu--emit (nelisp-gpu--arith-op (car e) ty)
                        (list (nelisp-gpu--type ty) id (car a) (car b)))
      (cons id ty)))
   ((eq (car-safe e) 'exp)
    (let* ((a (nelisp-gpu--lower (nth 1 e))) (id (nelisp-gpu--newid)))
      (nelisp-gpu--emit 12 (list (nelisp-gpu--type :float) id
                                 nelisp-gpu--glsl 27 (car a)))  ; GLSL Exp
      (cons id :float)))
   ((eq (car-safe e) 'sqrt)
    (let* ((a (nelisp-gpu--lower (nth 1 e))) (id (nelisp-gpu--newid)))
      (nelisp-gpu--emit 12 (list (nelisp-gpu--type :float) id
                                 nelisp-gpu--glsl 31 (car a)))  ; GLSL Sqrt
      (cons id :float)))
   ((eq (car-safe e) 'float)
    (let ((a (nelisp-gpu--lower (nth 1 e))))
      (cond ((eq (cdr a) :float) a)
            ((eq (cdr a) :int)
             (let ((id (nelisp-gpu--newid)))
               (nelisp-gpu--emit 111 (list (nelisp-gpu--type :float) id (car a))) ; ConvertSToF
               (cons id :float)))
            (t (let ((id (nelisp-gpu--newid)))
                 (nelisp-gpu--emit 112 (list (nelisp-gpu--type :float) id (car a))) ; ConvertUToF
                 (cons id :float))))))
   ((eq (car-safe e) 'sdot)               ; OpSDot of two 4x8-bit-packed uint32 -> int32
    (let* ((a (nelisp-gpu--lower (nth 1 e))) (b (nelisp-gpu--lower (nth 2 e)))
           (id (nelisp-gpu--newid)))
      (nelisp-gpu--emit 4450 (list (nelisp-gpu--type :int) id (car a) (car b) 1)) ; PackedVectorFormat4x8Bit
      (cons id :int)))
   ((eq (car-safe e) 'bitcast-u)          ; reinterpret a float's 32 bits as uint32 (OpBitcast)
    (let* ((a (nelisp-gpu--lower (nth 1 e))) (id (nelisp-gpu--newid)))
      (nelisp-gpu--emit 124 (list (nelisp-gpu--type :uint) id (car a)))
      (cons id :uint)))
   ((eq (car-safe e) 'uint)
    (let ((a (nelisp-gpu--lower (nth 1 e))))
      (if (eq (cdr a) :uint) a
        (let ((id (nelisp-gpu--newid)))
          (nelisp-gpu--emit 110 (list (nelisp-gpu--type :uint) id (car a))) ; ConvertFToU
          (cons id :uint)))))
   ((memq (car-safe e) '(< <= > >= =))
    (let* ((a (nelisp-gpu--lower (nth 1 e))) (b (nelisp-gpu--lower (nth 2 e)))
           (id (nelisp-gpu--newid)))
      (nelisp-gpu--emit (nelisp-gpu--cmp-op (car e) (eq (cdr a) :float))
                        (list (nelisp-gpu--type :bool) id (car a) (car b)))
      (cons id :bool)))
   (t (error "nelisp-gpu: cannot lower %S" e))))

;; --- statement lowering ----------------------------------------------
(defvar nelisp-gpu--bufs)   ; alist NAME -> variable id
(defvar nelisp-gpu--pcvar)  ; push-constant variable id
(defvar nelisp-gpu--lid)    ; LocalInvocationId builtin var id
(defvar nelisp-gpu--wgid)   ; WorkgroupId builtin var id
(defvar nelisp-gpu--shared) ; alist NAME -> workgroup array var id

(defun nelisp-gpu--store-local (name val-id)
  (nelisp-gpu--emit 62 (list (cadr (assq name nelisp-gpu--env)) val-id)))

(defun nelisp-gpu--lower-stmt (s)
  (pcase (car s)
    (`declare
     (nelisp-gpu--store-local (nth 1 s) (car (nelisp-gpu--lower (nth 3 s)))))
    (`set
     (nelisp-gpu--store-local (nth 1 s) (car (nelisp-gpu--lower (nth 2 s)))))
    (`store
     (let* ((tgt (nth 1 s))            ; (aref BUF IDX)
            (bufvar (cdr (assq (nth 1 tgt) nelisp-gpu--bufs)))
            (idx (car (nelisp-gpu--lower (nth 2 tgt))))
            (val (car (nelisp-gpu--lower (nth 2 s))))
            (pid (nelisp-gpu--buffer-elt-ptr bufvar idx)))
       (nelisp-gpu--emit 62 (list pid val))))
    (`when
     (let* ((cond (car (nelisp-gpu--lower (nth 1 s))))
            (lthen (nelisp-gpu--newid)) (lmerge (nelisp-gpu--newid)))
       (nelisp-gpu--emit 247 (list lmerge 0))
       (nelisp-gpu--emit 250 (list cond lthen lmerge))
       (nelisp-gpu--emit 248 (list lthen))
       (dolist (b (cddr s)) (nelisp-gpu--lower-stmt b))
       (nelisp-gpu--emit 249 (list lmerge))
       (nelisp-gpu--emit 248 (list lmerge))))
    (`for
     (let* ((spec (nth 1 s)) (var (nth 0 spec))
            (lo (nth 1 spec)) (hi (nth 2 spec))
            (lh (nelisp-gpu--newid)) (lck (nelisp-gpu--newid))
            (lb (nelisp-gpu--newid)) (lc (nelisp-gpu--newid))
            (lm (nelisp-gpu--newid)))
       (nelisp-gpu--store-local var (car (nelisp-gpu--lower lo)))
       (nelisp-gpu--emit 249 (list lh))
       (nelisp-gpu--emit 248 (list lh))
       (nelisp-gpu--emit 246 (list lm lc 0))      ; LoopMerge merge continue None
       (nelisp-gpu--emit 249 (list lck))
       (nelisp-gpu--emit 248 (list lck))
       (let* ((kv (nelisp-gpu--lower var)) (hv (nelisp-gpu--lower hi))
              (c (nelisp-gpu--newid)))
         (nelisp-gpu--emit (nelisp-gpu--cmp-op '< nil)
                           (list (nelisp-gpu--type :bool) c (car kv) (car hv)))
         (nelisp-gpu--emit 250 (list c lb lm)))
       (nelisp-gpu--emit 248 (list lb))
       (dolist (b (cddr s)) (nelisp-gpu--lower-stmt b))
       (nelisp-gpu--emit 249 (list lc))
       (nelisp-gpu--emit 248 (list lc))
       (let* ((kv (nelisp-gpu--lower var)) (id (nelisp-gpu--newid)))
         (nelisp-gpu--emit 128 (list (nelisp-gpu--type :uint) id
                                     (car kv) (nelisp-gpu--const-uint 1)))
         (nelisp-gpu--store-local var id))
       (nelisp-gpu--emit 249 (list lh))
       (nelisp-gpu--emit 248 (list lm))))
    (`store-shared
     (let* ((sv (cdr (assq (nth 1 s) nelisp-gpu--shared)))
            (idx (car (nelisp-gpu--lower (nth 2 s))))
            (val (car (nelisp-gpu--lower (nth 3 s))))
            (pid (nelisp-gpu--newid)))
       (unless sv (error "nelisp-gpu: unknown shared array %S" (nth 1 s)))
       (nelisp-gpu--emit 65 (list (nelisp-gpu--type '(:ptr 4 :float)) pid sv idx))
       (nelisp-gpu--emit 62 (list pid val))))
    (`barrier
     (nelisp-gpu--emit 224 (list (nelisp-gpu--const-uint 2)
                                 (nelisp-gpu--const-uint 2)
                                 (nelisp-gpu--const-uint 264))))
    (_ (error "nelisp-gpu: unknown stmt %S" s))))

(defun nelisp-gpu--collect-locals (stmts)
  "Walk STMTS collecting (NAME . TYPE) for every `declare' and `for'."
  (let (acc)
    (dolist (s stmts)
      (pcase (car s)
        (`declare (push (cons (nth 1 s) (nth 2 s)) acc))
        (`for (push (cons (nth 0 (nth 1 s)) :uint) acc)
              (setq acc (append (nelisp-gpu--collect-locals (cddr s)) acc)))
        (`when (setq acc (append (nelisp-gpu--collect-locals (cddr s)) acc)))))
    (nreverse acc)))

;; --- top level --------------------------------------------------------
(defun nelisp-gpu--uses-sym (form sym)
  "Non-nil if SYM appears as a call head anywhere in FORM."
  (cond ((and (consp form) (eq (car form) sym)) t)
        ((consp form) (or (nelisp-gpu--uses-sym (car form) sym)
                          (nelisp-gpu--uses-sym (cdr form) sym)))
        (t nil)))

(defun nelisp-gpu-compile (spec)
  "Compile kernel SPEC (plist) to a SPIR-V word list."
  (let* ((buffers (plist-get spec :buffers))
         (usesdot (nelisp-gpu--uses-sym (plist-get spec :body) 'sdot))
         (push (plist-get spec :push))
         (local (or (plist-get spec :local-size) 64))
         (body (plist-get spec :body))
         (nelisp-gpu--next 1)
         (nelisp-gpu--decos nil) (nelisp-gpu--globals nil)
         (nelisp-gpu--cache (make-hash-table :test 'equal))
         (nelisp-gpu--locals nil) (nelisp-gpu--body nil)
         (nelisp-gpu--env nil) (nelisp-gpu--bufs nil)
         (nelisp-gpu--push nil) (nelisp-gpu--pcvar nil)
         (nelisp-gpu--gid nil) (nelisp-gpu--main nil) (nelisp-gpu--glsl nil)
         (nelisp-gpu--lid nil) (nelisp-gpu--wgid nil) (nelisp-gpu--shared nil))
    (setq nelisp-gpu--main (nelisp-gpu--newid))
    (setq nelisp-gpu--gid (nelisp-gpu--newid))
    (setq nelisp-gpu--glsl (nelisp-gpu--newid))
    ;; gid variable (Input v3uint), BuiltIn GlobalInvocationId
    (nelisp-gpu--global 59 (list (nelisp-gpu--type '(:ptr 1 :v3uint))
                                 nelisp-gpu--gid 1))
    (nelisp-gpu--deco 71 (list nelisp-gpu--gid 11 28))
    ;; LocalInvocationId / WorkgroupId only when shared memory is used
    ;; (unused builtins crash some drivers; keep simple kernels minimal).
    (when (plist-get spec :shared)
      (setq nelisp-gpu--lid (nelisp-gpu--newid))
      (setq nelisp-gpu--wgid (nelisp-gpu--newid))
      (nelisp-gpu--global 59 (list (nelisp-gpu--type '(:ptr 1 :v3uint)) nelisp-gpu--lid 1))
      (nelisp-gpu--deco 71 (list nelisp-gpu--lid 11 27))
      (nelisp-gpu--global 59 (list (nelisp-gpu--type '(:ptr 1 :v3uint)) nelisp-gpu--wgid 1))
      (nelisp-gpu--deco 71 (list nelisp-gpu--wgid 11 26)))
    ;; storage buffers
    (let ((i 0) (ssboptr (nelisp-gpu--type '(:ptr 2 :ssbo))))
      (dolist (b buffers)
        (let ((v (nelisp-gpu--newid)))
          (nelisp-gpu--global 59 (list ssboptr v 2))
          (nelisp-gpu--deco 71 (list v 34 0))     ; DescriptorSet 0
          (nelisp-gpu--deco 71 (list v 33 i))     ; Binding i
          (push (cons b v) nelisp-gpu--bufs))
        (cl-incf i)))
    ;; push-constant block
    (when push
      (let* ((pcs (nelisp-gpu--type (list :pcstruct (length push))))
             (ptr (nelisp-gpu--type (list :ptrid 9 pcs)))
             (v (nelisp-gpu--newid)) (i 0))
        (nelisp-gpu--global 59 (list ptr v 9))
        (setq nelisp-gpu--pcvar v)
        (dolist (p push) (push (cons p i) nelisp-gpu--push) (cl-incf i))))
    ;; workgroup shared float arrays
    (dolist (sh (plist-get spec :shared))
      (let ((v (nelisp-gpu--newid)))
        (nelisp-gpu--global 59 (list (nelisp-gpu--type (list :ptr-wg-arr (nth 1 sh))) v 4))
        (push (cons (nth 0 sh) v) nelisp-gpu--shared)))
    ;; function-local variables (declared at function top)
    (let ((fptr-u (nelisp-gpu--type '(:ptr 7 :uint)))
          (fptr-f (nelisp-gpu--type '(:ptr 7 :float)))
          (fptr-i nil))
      (dolist (nt (nelisp-gpu--collect-locals body))
        (let ((id (nelisp-gpu--newid)) (ty (cdr nt)))
          (when (and (eq ty :int) (not fptr-i)) (setq fptr-i (nelisp-gpu--type '(:ptr 7 :int))))
          (push (nelisp-gpu--w 59 (list (cond ((eq ty :float) fptr-f) ((eq ty :int) fptr-i) (t fptr-u)) id 7))
                nelisp-gpu--locals)
          (push (cons (car nt) (cons id ty)) nelisp-gpu--env))))
    ;; function body
    (let ((lentry (nelisp-gpu--newid)))
      (dolist (s body) (nelisp-gpu--lower-stmt s))
      ;; assemble: header + caps + entry + execmode + decos + globals
      ;;           + function(label, locals, body, return)
      (let* ((fnty (nelisp-gpu--type :fnvoid))
             (voidty (nelisp-gpu--type :void))
             (bound nelisp-gpu--next)
             (out nil))
        (cl-flet ((add (ws) (setq out (append out ws))))
          (add (list #x07230203 #x00010000 0 bound 0))
          (add (nelisp-gpu--w 17 (list 1)))                 ; Capability Shader
          (when usesdot
            (add (nelisp-gpu--w 17 (list 6019)))            ; Capability DotProductKHR
            (add (nelisp-gpu--w 17 (list 6017)))            ; Capability DotProductInput4x8BitPackedKHR
            (add (nelisp-gpu--w 10 (nelisp-gpu--string-words "SPV_KHR_integer_dot_product"))))
          (add (nelisp-gpu--w 11 (cons nelisp-gpu--glsl     ; ExtInstImport
                                       (nelisp-gpu--string-words "GLSL.std.450"))))
          (add (nelisp-gpu--w 14 (list 0 1)))               ; MemoryModel
          (add (nelisp-gpu--w 15 (append (list 5 nelisp-gpu--main)
                                         (nelisp-gpu--string-words "main")
                                         (if nelisp-gpu--lid
                                             (list nelisp-gpu--gid nelisp-gpu--lid
                                                   nelisp-gpu--wgid)
                                           (list nelisp-gpu--gid)))))
          (let ((lx (if (consp local) (nth 0 local) local))
                (ly (if (consp local) (nth 1 local) 1)))
            (add (nelisp-gpu--w 16 (list nelisp-gpu--main 17 lx ly 1))))
          (dolist (d (nreverse nelisp-gpu--decos)) (add d))
          (dolist (g (nreverse nelisp-gpu--globals)) (add g))
          (add (nelisp-gpu--w 54 (list voidty nelisp-gpu--main 0 fnty)))
          (add (nelisp-gpu--w 248 (list lentry)))
          (dolist (l (nreverse nelisp-gpu--locals)) (add l))
          (dolist (b (nreverse nelisp-gpu--body)) (add b))
          (add (nelisp-gpu--w 253 nil))                     ; Return
          (add (nelisp-gpu--w 56 nil)))                     ; FunctionEnd
        out))))

(defun nelisp-gpu-compile-to-file (spec path)
  "Compile kernel SPEC and write the SPIR-V binary to PATH."
  (let ((words (nelisp-gpu-compile spec)))
    (with-temp-buffer
      (set-buffer-multibyte nil)
      (dolist (wd words)
        (insert (logand wd #xff) (logand (ash wd -8) #xff)
                (logand (ash wd -16) #xff) (logand (ash wd -24) #xff)))
      (let ((coding-system-for-write 'binary))
        (write-region (point-min) (point-max) path nil 'silent)))
    path))

(provide 'nelisp-gpu-compile)
;;; nelisp-gpu-compile.el ends here
