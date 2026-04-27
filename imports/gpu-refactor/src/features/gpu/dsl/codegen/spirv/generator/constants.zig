const std = @import("std");

pub const SPIRV_MAGIC: u32 = 0x07230203;
pub const SPIRV_VERSION: u32 = 0x00010000;
pub const GENERATOR_ID: u32 = 0x12340000;

pub const OpCode = enum(u32) {
    OpNop = 0,
    OpSource = 3,
    OpName = 5,
    OpMemberName = 6,
    OpExtension = 10,
    OpExtInstImport = 11,
    OpExtInst = 12,
    OpMemoryModel = 14,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpCapability = 17,
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeArray = 24,
    OpTypeRuntimeArray = 25,
    OpTypeStruct = 26,
    OpTypePointer = 32,
    OpTypeFunction = 33,
    OpConstantTrue = 37,
    OpConstantFalse = 38,
    OpConstant = 43,
    OpFunction = 54,
    OpFunctionEnd = 56,
    OpVariable = 59,
    OpLoad = 61,
    OpStore = 62,
    OpAccessChain = 65,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpFNegate = 127,
    OpFAdd = 129,
    OpFSub = 131,
    OpFMul = 133,
    OpFDiv = 136,
    OpFMod = 139,
    OpIAdd = 128,
    OpISub = 130,
    OpIMul = 132,
    OpSDiv = 135,
    OpUDiv = 134,
    OpSRem = 138,
    OpUMod = 137,
    OpIEqual = 169,
    OpINotEqual = 170,
    OpSLessThan = 177,
    OpULessThan = 176,
    OpSLessThanEqual = 179,
    OpULessThanEqual = 178,
    OpSGreaterThan = 173,
    OpUGreaterThan = 172,
    OpSGreaterThanEqual = 175,
    OpUGreaterThanEqual = 174,
    OpFOrdEqual = 180,
    OpFOrdNotEqual = 182,
    OpFOrdLessThan = 184,
    OpFOrdLessThanEqual = 186,
    OpFOrdGreaterThan = 188,
    OpFOrdGreaterThanEqual = 190,
    OpLogicalEqual = 164,
    OpLogicalNotEqual = 165,
    OpLogicalAnd = 167,
    OpLogicalOr = 168,
    OpLogicalNot = 166,
    OpBitwiseAnd = 194,
    OpBitwiseOr = 195,
    OpBitwiseXor = 196,
    OpNot = 197,
    OpShiftLeftLogical = 198,
    OpShiftRightLogical = 199,
    OpShiftRightArithmetic = 200,
    OpSelect = 191,
    OpCompositeConstruct = 80,
    OpCompositeExtract = 81,
    OpVectorShuffle = 79,
    OpBitcast = 124,
    OpLabel = 248,
    OpBranch = 249,
    OpBranchConditional = 250,
    OpSwitch = 251,
    OpKill = 252,
    OpReturn = 253,
    OpReturnValue = 254,
    OpUnreachable = 255,
    OpLoopMerge = 246,
    OpSelectionMerge = 247,
    OpControlBarrier = 224,
    OpMemoryBarrier = 225,
    OpAtomicIAdd = 227,
};

pub const Capability = enum(u32) {
    Shader = 1,
    Float64 = 12,
    Int64 = 11,
    GroupNonUniform = 61,
};

pub const ExecutionModel = enum(u32) {
    GLCompute = 5,
};

pub const AddressingModel = enum(u32) {
    Logical = 0,
};

pub const MemoryModel = enum(u32) {
    GLSL450 = 1,
};

pub const ExecutionMode = enum(u32) {
    LocalSize = 17,
};

pub const StorageClass = enum(u32) {
    Uniform = 2,
    Input = 1,
    Workgroup = 4,
    StorageBuffer = 12,
    Function = 7,
};

pub const Decoration = enum(u32) {
    Block = 2,
    Offset = 35,
    ArrayStride = 6,
    DescriptorSet = 34,
    Binding = 33,
    BuiltIn = 11,
    NonWritable = 24,
};

pub const BuiltIn = enum(u32) {
    GlobalInvocationId = 28,
    LocalInvocationId = 27,
    WorkgroupId = 26,
    LocalInvocationIndex = 29,
};

pub const MemorySemantics = enum(u32) {
    None = 0,
    AcquireRelease = 0x8,
    WorkgroupMemory = 0x100,
};

pub const Scope = enum(u32) {
    Device = 1,
    Workgroup = 2,
};
