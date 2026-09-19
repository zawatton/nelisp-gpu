import struct
W=[]
def w(*xs):
    for x in xs: W.append(x & 0xffffffff)
def ins(op, *ops):
    wc=len(ops)+1
    w((wc<<16)|op, *ops)
def s(text):  # packed, null-terminated string -> list of words
    bs=text.encode()+b'\x00'
    while len(bs)%4: bs+=b'\x00'
    return list(struct.unpack("<%dI"%(len(bs)//4), bs))
# ---- ids ----
(main,gid,void,fnvoid,uint,flt,v3uint,p_in_v3,p_in_u,u0,rtarr,sstruct,
 p_u_ss,varA,varB,varC,p_u_f,pcstruct,p_pc,varPC,p_pc_u,boolt,
 lentry,gv,i_,npp,n_,cond,lthen,lmerge,pa,va,pb,vb,summ,pcp)=range(1,37)
BOUND=37
# ---- header ----
w(0x07230203, 0x00010000, 0, BOUND, 0)
ins(17,1)                      # OpCapability Shader
ins(14,0,1)                    # OpMemoryModel Logical GLSL450
ins(15, 5, main, *s("main"), gid)   # OpEntryPoint GLCompute %main "main" %gid
ins(16, main, 17, 64,1,1)      # OpExecutionMode LocalSize 64 1 1
# ---- decorations ----
ins(71, gid, 11, 28)           # Decorate gid BuiltIn GlobalInvocationId
ins(71, rtarr, 6, 4)           # Decorate rtarr ArrayStride 4
ins(71, sstruct, 3)            # Decorate sstruct BufferBlock
ins(72, sstruct, 0, 35, 0)     # MemberDecorate sstruct 0 Offset 0
ins(71, varA, 34, 0); ins(71, varA, 33, 0)  # DescriptorSet 0, Binding 0
ins(71, varB, 34, 0); ins(71, varB, 33, 1)
ins(71, varC, 34, 0); ins(71, varC, 33, 2)
ins(71, pcstruct, 2)           # Decorate pcstruct Block
ins(72, pcstruct, 0, 35, 0)    # MemberDecorate pcstruct 0 Offset 0
# ---- types/consts/vars ----
ins(19, void)                  # OpTypeVoid
ins(33, fnvoid, void)          # OpTypeFunction void
ins(21, uint, 32, 0)           # OpTypeInt 32 unsigned
ins(22, flt, 32)               # OpTypeFloat 32
ins(23, v3uint, uint, 3)       # OpTypeVector uint 3
ins(32, p_in_v3, 1, v3uint)    # OpTypePointer Input v3uint
ins(32, p_in_u, 1, uint)       # OpTypePointer Input uint
ins(43, uint, u0, 0)           # OpConstant uint 0
ins(59, p_in_v3, gid, 1)       # OpVariable p_in_v3 Input  (gid)
ins(29, rtarr, flt)            # OpTypeRuntimeArray float
ins(30, sstruct, rtarr)        # OpTypeStruct { rtarr }
ins(32, p_u_ss, 2, sstruct)    # OpTypePointer Uniform sstruct
ins(59, p_u_ss, varA, 2)       # OpVariable Uniform (A)
ins(59, p_u_ss, varB, 2)
ins(59, p_u_ss, varC, 2)
ins(32, p_u_f, 2, flt)         # OpTypePointer Uniform float
ins(30, pcstruct, uint)        # OpTypeStruct { uint }
ins(32, p_pc, 9, pcstruct)     # OpTypePointer PushConstant pcstruct
ins(59, p_pc, varPC, 9)        # OpVariable PushConstant
ins(32, p_pc_u, 9, uint)       # OpTypePointer PushConstant uint
ins(20, boolt)                 # OpTypeBool
# ---- function ----
ins(54, void, main, 0, fnvoid) # OpFunction void None fnvoid
ins(248, lentry)               # OpLabel
ins(61, v3uint, gv, gid)       # OpLoad v3uint gid
ins(81, uint, i_, gv, 0)       # OpCompositeExtract uint gv 0
ins(65, p_pc_u, npp, varPC, u0)# OpAccessChain pushconst.member0
ins(61, uint, n_, npp)         # OpLoad uint n
ins(176, boolt, cond, i_, n_)  # OpULessThan i < n
ins(247, lmerge, 0)            # OpSelectionMerge merge None
ins(250, cond, lthen, lmerge)  # OpBranchConditional
ins(248, lthen)                # OpLabel then
ins(65, p_u_f, pa, varA, u0, i_); ins(61, flt, va, pa)
ins(65, p_u_f, pb, varB, u0, i_); ins(61, flt, vb, pb)
ins(129, flt, summ, va, vb)    # OpFAdd
ins(65, p_u_f, pcp, varC, u0, i_); ins(62, pcp, summ)  # OpStore
ins(249, lmerge)               # OpBranch merge
ins(248, lmerge)               # OpLabel merge
ins(253)                       # OpReturn
ins(56)                        # OpFunctionEnd
data=struct.pack("<%dI"%len(W), *W)
open("kernels/vadd_hand.spv","wb").write(data)
print("wrote kernels/vadd_hand.spv bytes=%d words=%d"%(len(data),len(W)))
