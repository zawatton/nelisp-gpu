;;; nelisp-gpu-spirv.el --- Emit SPIR-V compute modules from elisp  -*- lexical-binding: t; -*-

;; The kernel compiler's lowest layer: assemble a SPIR-V binary module
;; word-by-word from elisp and write it to disk.  Proves "GPU kernels
;; are written in elisp": the .spv produced here runs as native GPU code
;; under Vulkan (see host/vkcompute).  This file hand-builds the vadd
;; (c = a + b) module; later layers turn an s-expression kernel DSL into
;; this same word stream.

;;; Code:

;; ---- low-level word/byte emission --------------------------------------

(defun nelisp-gpu-spirv--string-words (str)
  "Pack STR as SPIR-V literal: null-terminated, 0-padded to 4-byte words."
  (let ((bytes (append (string-to-list str) (list 0))) words)
    (while (/= (% (length bytes) 4) 0) (setq bytes (append bytes (list 0))))
    (let ((b bytes))
      (while b
        (push (logior (nth 0 b) (ash (nth 1 b) 8)
                      (ash (nth 2 b) 16) (ash (nth 3 b) 24))
              words)
        (setq b (nthcdr 4 b))))
    (nreverse words)))

(defun nelisp-gpu-spirv--write (words path)
  "Write little-endian uint32 WORDS to PATH as a raw binary file."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (dolist (wd words)
      (insert (logand wd #xff)
              (logand (ash wd -8) #xff)
              (logand (ash wd -16) #xff)
              (logand (ash wd -24) #xff)))
    (let ((coding-system-for-write 'binary))
      (write-region (point-min) (point-max) path nil 'silent)))
  path)

;; A tiny instruction-stream builder.  `ins' appends one SPIR-V
;; instruction (word count is computed); `wraw' appends literal words.
(defun nelisp-gpu-spirv--make-stream () (list :words nil))
(defun nelisp-gpu-spirv--wraw (st &rest ws)
  (plist-put st :words (append (plist-get st :words) ws)))
(defun nelisp-gpu-spirv--ins (st op &rest operands)
  (let ((wc (1+ (length operands))))
    (plist-put st :words
               (append (plist-get st :words)
                       (cons (logior (ash wc 16) op) operands)))))

;; ---- the vadd (c = a + b) compute module -------------------------------

(defun nelisp-gpu-spirv-vadd-words ()
  "Return the SPIR-V word list for the vadd compute kernel.
Interface matches host/vkcompute: set 0 bindings 0/1/2 storage buffers
\(a, b, c), a push-constant uint n, local_size_x = 64."
  ;; explicit result-id assignment (1..36), bound = 37
  (let* ((main 1) (gid 2) (void 3) (fnvoid 4) (uint 5) (flt 6) (v3uint 7)
         (p-in-v3 8) (p-in-u 9) (u0 10) (rtarr 11) (sstruct 12) (p-u-ss 13)
         (varA 14) (varB 15) (varC 16) (p-u-f 17) (pcstruct 18) (p-pc 19)
         (varPC 20) (p-pc-u 21) (boolt 22) (lentry 23) (gv 24) (i_ 25)
         (npp 26) (n_ 27) (cond 28) (lthen 29) (lmerge 30) (pa 31) (va 32)
         (pb 33) (vb 34) (summ 35) (pcp 36) (bound 37)
         (st (nelisp-gpu-spirv--make-stream)))
    ;; module header: magic, version 1.0, generator 0, bound, schema 0
    (nelisp-gpu-spirv--wraw st #x07230203 #x00010000 0 bound 0)
    (nelisp-gpu-spirv--ins st 17 1)            ; Capability Shader
    (nelisp-gpu-spirv--ins st 14 0 1)          ; MemoryModel Logical GLSL450
    (apply #'nelisp-gpu-spirv--ins st 15 5 main
           (append (nelisp-gpu-spirv--string-words "main") (list gid))) ; EntryPoint
    (nelisp-gpu-spirv--ins st 16 main 17 64 1 1) ; ExecutionMode LocalSize 64 1 1
    ;; decorations
    (nelisp-gpu-spirv--ins st 71 gid 11 28)    ; gid BuiltIn GlobalInvocationId
    (nelisp-gpu-spirv--ins st 71 rtarr 6 4)    ; rtarr ArrayStride 4
    (nelisp-gpu-spirv--ins st 71 sstruct 3)    ; sstruct BufferBlock
    (nelisp-gpu-spirv--ins st 72 sstruct 0 35 0) ; member0 Offset 0
    (nelisp-gpu-spirv--ins st 71 varA 34 0) (nelisp-gpu-spirv--ins st 71 varA 33 0)
    (nelisp-gpu-spirv--ins st 71 varB 34 0) (nelisp-gpu-spirv--ins st 71 varB 33 1)
    (nelisp-gpu-spirv--ins st 71 varC 34 0) (nelisp-gpu-spirv--ins st 71 varC 33 2)
    (nelisp-gpu-spirv--ins st 71 pcstruct 2)   ; pcstruct Block
    (nelisp-gpu-spirv--ins st 72 pcstruct 0 35 0) ; member0 Offset 0
    ;; types / constants / variables
    (nelisp-gpu-spirv--ins st 19 void)         ; TypeVoid
    (nelisp-gpu-spirv--ins st 33 fnvoid void)  ; TypeFunction void
    (nelisp-gpu-spirv--ins st 21 uint 32 0)    ; TypeInt 32 unsigned
    (nelisp-gpu-spirv--ins st 22 flt 32)       ; TypeFloat 32
    (nelisp-gpu-spirv--ins st 23 v3uint uint 3) ; TypeVector uint 3
    (nelisp-gpu-spirv--ins st 32 p-in-v3 1 v3uint) ; Pointer Input v3uint
    (nelisp-gpu-spirv--ins st 32 p-in-u 1 uint)    ; Pointer Input uint
    (nelisp-gpu-spirv--ins st 43 uint u0 0)    ; Constant uint 0
    (nelisp-gpu-spirv--ins st 59 p-in-v3 gid 1) ; Variable Input (gid)
    (nelisp-gpu-spirv--ins st 29 rtarr flt)    ; TypeRuntimeArray float
    (nelisp-gpu-spirv--ins st 30 sstruct rtarr) ; TypeStruct {rtarr}
    (nelisp-gpu-spirv--ins st 32 p-u-ss 2 sstruct) ; Pointer Uniform sstruct
    (nelisp-gpu-spirv--ins st 59 p-u-ss varA 2) ; Variable Uniform A
    (nelisp-gpu-spirv--ins st 59 p-u-ss varB 2)
    (nelisp-gpu-spirv--ins st 59 p-u-ss varC 2)
    (nelisp-gpu-spirv--ins st 32 p-u-f 2 flt)  ; Pointer Uniform float
    (nelisp-gpu-spirv--ins st 30 pcstruct uint) ; TypeStruct {uint}
    (nelisp-gpu-spirv--ins st 32 p-pc 9 pcstruct) ; Pointer PushConstant
    (nelisp-gpu-spirv--ins st 59 p-pc varPC 9) ; Variable PushConstant
    (nelisp-gpu-spirv--ins st 32 p-pc-u 9 uint) ; Pointer PushConstant uint
    (nelisp-gpu-spirv--ins st 20 boolt)        ; TypeBool
    ;; function body
    (nelisp-gpu-spirv--ins st 54 void main 0 fnvoid) ; Function void None fnvoid
    (nelisp-gpu-spirv--ins st 248 lentry)      ; Label
    (nelisp-gpu-spirv--ins st 61 v3uint gv gid) ; Load gid
    (nelisp-gpu-spirv--ins st 81 uint i_ gv 0) ; CompositeExtract .x
    (nelisp-gpu-spirv--ins st 65 p-pc-u npp varPC u0) ; AccessChain push.n
    (nelisp-gpu-spirv--ins st 61 uint n_ npp)  ; Load n
    (nelisp-gpu-spirv--ins st 176 boolt cond i_ n_) ; ULessThan i<n
    (nelisp-gpu-spirv--ins st 247 lmerge 0)    ; SelectionMerge
    (nelisp-gpu-spirv--ins st 250 cond lthen lmerge) ; BranchConditional
    (nelisp-gpu-spirv--ins st 248 lthen)       ; Label then
    (nelisp-gpu-spirv--ins st 65 p-u-f pa varA u0 i_) (nelisp-gpu-spirv--ins st 61 flt va pa)
    (nelisp-gpu-spirv--ins st 65 p-u-f pb varB u0 i_) (nelisp-gpu-spirv--ins st 61 flt vb pb)
    (nelisp-gpu-spirv--ins st 129 flt summ va vb) ; FAdd
    (nelisp-gpu-spirv--ins st 65 p-u-f pcp varC u0 i_) (nelisp-gpu-spirv--ins st 62 pcp summ) ; Store
    (nelisp-gpu-spirv--ins st 249 lmerge)      ; Branch merge
    (nelisp-gpu-spirv--ins st 248 lmerge)      ; Label merge
    (nelisp-gpu-spirv--ins st 253)             ; Return
    (nelisp-gpu-spirv--ins st 56)              ; FunctionEnd
    (plist-get st :words)))

(defun nelisp-gpu-spirv-write-vadd (path)
  "Emit the vadd SPIR-V module to PATH."
  (nelisp-gpu-spirv--write (nelisp-gpu-spirv-vadd-words) path))

;; --- hand-emitted matmul with a Phi (register) accumulator ----------
;; Tests whether an SSA/register accumulator (no per-iteration memory
;; load/store of the running sum) lifts matmul throughput at 1024^3.
;; Buffers A(M,K) B(K,N) C(M,N), push (M K N), local 64, naive dispatch.
(defun nelisp-gpu-spirv-matmul-phi-words ()
  (let* ((main 1) (gid 2) (void 3) (fnvoid 4) (uint 5) (flt 6) (v3 7) (pInV3 8)
         (u0 9) (u1 10) (u2 11) (f0 12) (rtarr 13) (ssbo 14) (pUss 15)
         (A 16) (B 17) (C 18) (pUf 19) (pcs 20) (pPc 21) (varPC 22) (pPcU 23)
         (boolt 24) (entry 25) (gv 26) (idx 27) (pM 28) (M 29) (pK 30) (K 31)
         (pN 32) (N 33) (mn 34) (cond0 35) (then 36) (end 37) (row 38) (col 39)
         (rowK 40) (loop 41) (acc 42) (k 43) (lmerge 44) (cont 45) (lcheck 46)
         (c 47) (lbody 48) (aidx 49) (pa 50) (av 51) (kN 52) (bidx 53) (pb 54)
         (bv 55) (prod 56) (accn 57) (kn 58) (pcp 59) (bound 60)
         (st (nelisp-gpu-spirv--make-stream)))
    (nelisp-gpu-spirv--wraw st #x07230203 #x00010000 0 bound 0)
    (nelisp-gpu-spirv--ins st 17 1)
    (nelisp-gpu-spirv--ins st 14 0 1)
    (apply #'nelisp-gpu-spirv--ins st 15 5 main
           (append (nelisp-gpu-spirv--string-words "main") (list gid)))
    (nelisp-gpu-spirv--ins st 16 main 17 64 1 1)
    (nelisp-gpu-spirv--ins st 71 gid 11 28)
    (nelisp-gpu-spirv--ins st 71 rtarr 6 4)
    (nelisp-gpu-spirv--ins st 71 ssbo 3)
    (nelisp-gpu-spirv--ins st 72 ssbo 0 35 0)
    (nelisp-gpu-spirv--ins st 71 A 34 0) (nelisp-gpu-spirv--ins st 71 A 33 0)
    (nelisp-gpu-spirv--ins st 71 B 34 0) (nelisp-gpu-spirv--ins st 71 B 33 1)
    (nelisp-gpu-spirv--ins st 71 C 34 0) (nelisp-gpu-spirv--ins st 71 C 33 2)
    (nelisp-gpu-spirv--ins st 71 pcs 2)
    (nelisp-gpu-spirv--ins st 72 pcs 0 35 0)
    (nelisp-gpu-spirv--ins st 72 pcs 1 35 4)
    (nelisp-gpu-spirv--ins st 72 pcs 2 35 8)
    (nelisp-gpu-spirv--ins st 19 void)
    (nelisp-gpu-spirv--ins st 33 fnvoid void)
    (nelisp-gpu-spirv--ins st 21 uint 32 0)
    (nelisp-gpu-spirv--ins st 22 flt 32)
    (nelisp-gpu-spirv--ins st 23 v3 uint 3)
    (nelisp-gpu-spirv--ins st 32 pInV3 1 v3)
    (nelisp-gpu-spirv--ins st 43 uint u0 0)
    (nelisp-gpu-spirv--ins st 43 uint u1 1)
    (nelisp-gpu-spirv--ins st 43 uint u2 2)
    (nelisp-gpu-spirv--ins st 43 flt f0 0)
    (nelisp-gpu-spirv--ins st 59 pInV3 gid 1)
    (nelisp-gpu-spirv--ins st 29 rtarr flt)
    (nelisp-gpu-spirv--ins st 30 ssbo rtarr)
    (nelisp-gpu-spirv--ins st 32 pUss 2 ssbo)
    (nelisp-gpu-spirv--ins st 59 pUss A 2)
    (nelisp-gpu-spirv--ins st 59 pUss B 2)
    (nelisp-gpu-spirv--ins st 59 pUss C 2)
    (nelisp-gpu-spirv--ins st 32 pUf 2 flt)
    (nelisp-gpu-spirv--ins st 30 pcs uint uint uint)
    (nelisp-gpu-spirv--ins st 32 pPc 9 pcs)
    (nelisp-gpu-spirv--ins st 59 pPc varPC 9)
    (nelisp-gpu-spirv--ins st 32 pPcU 9 uint)
    (nelisp-gpu-spirv--ins st 20 boolt)
    (nelisp-gpu-spirv--ins st 54 void main 0 fnvoid)
    (nelisp-gpu-spirv--ins st 248 entry)
    (nelisp-gpu-spirv--ins st 61 v3 gv gid)
    (nelisp-gpu-spirv--ins st 81 uint idx gv 0)
    (nelisp-gpu-spirv--ins st 65 pPcU pM varPC u0) (nelisp-gpu-spirv--ins st 61 uint M pM)
    (nelisp-gpu-spirv--ins st 65 pPcU pK varPC u1) (nelisp-gpu-spirv--ins st 61 uint K pK)
    (nelisp-gpu-spirv--ins st 65 pPcU pN varPC u2) (nelisp-gpu-spirv--ins st 61 uint N pN)
    (nelisp-gpu-spirv--ins st 132 uint mn M N)
    (nelisp-gpu-spirv--ins st 176 boolt cond0 idx mn)
    (nelisp-gpu-spirv--ins st 247 end 0)
    (nelisp-gpu-spirv--ins st 250 cond0 then end)
    (nelisp-gpu-spirv--ins st 248 then)
    (nelisp-gpu-spirv--ins st 134 uint row idx N)
    (nelisp-gpu-spirv--ins st 137 uint col idx N)
    (nelisp-gpu-spirv--ins st 132 uint rowK row K)
    (nelisp-gpu-spirv--ins st 249 loop)
    (nelisp-gpu-spirv--ins st 248 loop)
    (nelisp-gpu-spirv--ins st 245 flt acc f0 then accn cont)
    (nelisp-gpu-spirv--ins st 245 uint k u0 then kn cont)
    (nelisp-gpu-spirv--ins st 246 lmerge cont 0)
    (nelisp-gpu-spirv--ins st 249 lcheck)
    (nelisp-gpu-spirv--ins st 248 lcheck)
    (nelisp-gpu-spirv--ins st 176 boolt c k K)
    (nelisp-gpu-spirv--ins st 250 c lbody lmerge)
    (nelisp-gpu-spirv--ins st 248 lbody)
    (nelisp-gpu-spirv--ins st 128 uint aidx rowK k)
    (nelisp-gpu-spirv--ins st 65 pUf pa A u0 aidx) (nelisp-gpu-spirv--ins st 61 flt av pa)
    (nelisp-gpu-spirv--ins st 132 uint kN k N)
    (nelisp-gpu-spirv--ins st 128 uint bidx kN col)
    (nelisp-gpu-spirv--ins st 65 pUf pb B u0 bidx) (nelisp-gpu-spirv--ins st 61 flt bv pb)
    (nelisp-gpu-spirv--ins st 133 flt prod av bv)
    (nelisp-gpu-spirv--ins st 129 flt accn acc prod)
    (nelisp-gpu-spirv--ins st 249 cont)
    (nelisp-gpu-spirv--ins st 248 cont)
    (nelisp-gpu-spirv--ins st 128 uint kn k u1)
    (nelisp-gpu-spirv--ins st 249 loop)
    (nelisp-gpu-spirv--ins st 248 lmerge)
    (nelisp-gpu-spirv--ins st 65 pUf pcp C u0 idx) (nelisp-gpu-spirv--ins st 62 pcp acc)
    (nelisp-gpu-spirv--ins st 249 end)
    (nelisp-gpu-spirv--ins st 248 end)
    (nelisp-gpu-spirv--ins st 253)
    (nelisp-gpu-spirv--ins st 56)
    (plist-get st :words)))

(defun nelisp-gpu-spirv-matmul-phi-write (path)
  "Emit the Phi-accumulator matmul SPIR-V to PATH."
  (nelisp-gpu-spirv--write (nelisp-gpu-spirv-matmul-phi-words) path))

(provide 'nelisp-gpu-spirv)
;;; nelisp-gpu-spirv.el ends here
