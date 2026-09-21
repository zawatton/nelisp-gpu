# nelisp-gpu -- elisp -> SPIR-V -> Vulkan compute on the GPU.
#
# The C host (host/vkcompute) is the reference runtime used to de-risk
# the GPU path; the long-term host is elisp/nelisp-cc.  GLSL kernels are
# the gold reference for the elisp SPIR-V emitter.

GLSLANG ?= glslangValidator
CC      ?= cc
EMACS   ?= emacs
N       ?= 1024
M       ?= 512
K       ?= 512
NN      ?= 512


# Byte-compilation.  Nothing here was ever compiled, so every one of these
# files ran interpreted -- including the float32 codec that every activation
# crosses on its way to and from the device.  Compiling it is worth 2.4x on
# that codec alone and 1.6x on a whole transformer block.  The rule is
# per-file so `make' recompiles only what changed, because a .elc that is
# older than its .el is still preferred by `load' and only warns.
GPU_ELC := $(patsubst %.el,%.elc,$(wildcard lisp/*.el))

lisp/%.elc: lisp/%.el
	$(EMACS) -Q --batch -L lisp \
	  --eval '(setq byte-compile-warnings (quote (not docstrings)))' \
	  -f batch-byte-compile $<

compile: $(GPU_ELC)

.PHONY: host shaders run derisk el-derisk matmul-derisk verify f32 clean tools-check server ternary-kernel upload-file

host: host/vkcompute

host/vkcompute: host/vkcompute.c
	$(CC) -O2 host/vkcompute.c -lvulkan -o $@

shaders: kernels/vadd.spv

kernels/%.spv: kernels/%.comp
	$(GLSLANG) -V --target-env vulkan1.1 $< -o $@

# Full de-risk: build host, compile reference shader, run on the GPU.
derisk: host shaders
	./host/vkcompute kernels/vadd.spv $(N)

run: host
	./host/vkcompute kernels/vadd.spv $(N)

# Phase 1 proof: emit the SPIR-V kernel FROM ELISP, then run it on the GPU.
el-derisk: host
	$(EMACS) -Q --batch -L lisp -l nelisp-gpu-spirv \
	  --eval '(nelisp-gpu-spirv-write-vadd "kernels/vadd_el.spv")'
	./host/vkcompute kernels/vadd_el.spv $(N)

host/vkmatmul: host/vkmatmul.c
	$(CC) -O2 host/vkmatmul.c -lvulkan -o $@

# Phase 2 proof: compile the matmul kernel from the elisp DSL and run on the GPU.
matmul-derisk: host/vkmatmul
	$(EMACS) -Q --batch -L lisp -l nelisp-gpu-kernels \
	  --eval '(nelisp-gpu-write-kernel (quote matmul) "kernels/matmul.spv")'
	./host/vkmatmul kernels/matmul.spv $(M) $(K) $(NN)

host/vkrun: host/vkrun.c
	$(CC) -O2 host/vkrun.c -lvulkan -o $@

# The persistent server `nelisp-gpu-server.el' talks to.  It had no rule here,
# so a fresh checkout could not build the binary that every resident-weight
# path depends on.
host/vkserver: host/vkserver.c
	$(CC) -O2 host/vkserver.c -lvulkan -o $@

server: host/vkserver

# Verify all DSL kernels on the GPU against the photon-tensor CPU oracle.
verify: host/vkrun
	$(EMACS) -Q --batch -L lisp -l test/verify.el

# The float32 wire encoding.  No GPU: it is arithmetic, and it is the layer
# every buffer and every shader constant passes through, so a defect here is
# invisible in kernel comparisons and shows up as a wrong answer instead.
f32:
	$(EMACS) -Q --batch -L lisp -l test/f32-test.el

ternary-kernel: host/vkserver
	$(EMACS) -Q --batch -L lisp -l test/ternary-kernel-test.el

upload-file: host/vkserver
	$(EMACS) -Q --batch -L lisp -l test/upload-file-test.el

tools-check:
	@command -v $(GLSLANG) >/dev/null && echo "glslang: OK" || echo "glslang: MISSING (sudo apt-get install -y glslang-tools spirv-tools)"
	@command -v spirv-val   >/dev/null && echo "spirv-val: OK" || echo "spirv-val: MISSING"
	@command -v spirv-dis   >/dev/null && echo "spirv-dis: OK" || echo "spirv-dis: MISSING"

clean:
	rm -f host/vkcompute host/vkmatmul host/vkrun host/vkserver kernels/*.spv
