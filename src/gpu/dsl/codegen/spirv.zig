//! SPIR-V Code Generator
//!
//! Generates SPIR-V binary bytecode from kernel IR for Vulkan compute shaders.
//! SPIR-V is a binary intermediate representation defined by Khronos.
//!
//! Specification: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html

const std = @import("std");
const types = @import("../types.zig");
const expr = @import("../expr.zig");
const stmt = @import("../stmt.zig");
const kernel = @import("../kernel.zig");
const backend = @import("backend.zig");
const gpu_backend = @import("../../backend.zig");

/// SPIR-V magic number identifying the binary format.
pub const SPIRV_MAGIC: u32 = 0x07230203;

/// SPIR-V version (1.5).
pub const SPIRV_VERSION: u32 = 0x00010500;

/// Generator ID (0 = unregistered).
pub const GENERATOR_ID: u32 = 0;

/// SPIR-V opcodes for compute shaders.
pub const OpCode = enum(u16) {
    // Core
    OpNop = 0,
    OpUndef = 1,
    OpSourceContinued = 2,
    OpSource = 3,
    OpSourceExtension = 4,
    OpName = 5,
    OpMemberName = 6,
    OpString = 7,
    OpLine = 8,

    // Extensions
    OpExtension = 10,
    OpExtInstImport = 11,
    OpExtInst = 12,

    // Mode Setting
    OpMemoryModel = 14,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpCapability = 17,

    // Type-Declaration
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeMatrix = 24,
    OpTypeImage = 25,
    OpTypeSampler = 26,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypeOpaque = 31,
    OpTypePointer = 32,
    OpTypeFunction = 33,

    // Constant-Creation
    OpConstantTrue = 41,
    OpConstantFalse = 42,
    OpConstant = 43,
    OpConstantComposite = 44,
    OpConstantNull = 46,

    // Memory
    OpVariable = 59,
    OpImageTexelPointer = 60,
    OpLoad = 61,
    OpStore = 62,
    OpCopyMemory = 63,
    OpAccessChain = 65,
    OpInBoundsAccessChain = 66,

    // Function
    OpFunction = 54,
    OpFunctionParameter = 55,
    OpFunctionEnd = 56,
    OpFunctionCall = 57,

    // Control Flow
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

    // Arithmetic (Float)
    OpFNegate = 127,
    OpFAdd = 129,
    OpFSub = 131,
    OpFMul = 133,
    OpFDiv = 136,
    OpFRem = 137,
    OpFMod = 138,

    // Arithmetic (Integer)
    OpIAdd = 128,
    OpISub = 130,
    OpIMul = 132,
    OpUDiv = 134,
    OpSDiv = 135,
    OpUMod = 139,
    OpSMod = 140,
    OpSRem = 141,
    OpSNegate = 126,

    // Logical
    OpLogicalEqual = 164,
    OpLogicalNotEqual = 165,
    OpLogicalOr = 166,
    OpLogicalAnd = 167,
    OpLogicalNot = 168,
    OpSelect = 169,

    // Comparison (Integer)
    OpIEqual = 170,
    OpINotEqual = 171,
    OpUGreaterThan = 172,
    OpSGreaterThan = 173,
    OpUGreaterThanEqual = 174,
    OpSGreaterThanEqual = 175,
    OpULessThan = 176,
    OpSLessThan = 177,
    OpULessThanEqual = 178,
    OpSLessThanEqual = 179,

    // Comparison (Float)
    OpFOrdEqual = 180,
    OpFUnordEqual = 181,
    OpFOrdNotEqual = 182,
    OpFUnordNotEqual = 183,
    OpFOrdLessThan = 184,
    OpFUnordLessThan = 185,
    OpFOrdGreaterThan = 186,
    OpFUnordGreaterThan = 187,
    OpFOrdLessThanEqual = 188,
    OpFUnordLessThanEqual = 189,
    OpFOrdGreaterThanEqual = 190,
    OpFUnordGreaterThanEqual = 191,

    // Bitwise
    OpShiftRightLogical = 194,
    OpShiftRightArithmetic = 195,
    OpShiftLeftLogical = 196,
    OpBitwiseOr = 197,
    OpBitwiseXor = 198,
    OpBitwiseAnd = 199,
    OpNot = 200,

    // Conversion
    OpConvertFToU = 109,
    OpConvertFToS = 110,
    OpConvertSToF = 111,
    OpConvertUToF = 112,
    OpUConvert = 113,
    OpSConvert = 114,
    OpFConvert = 115,
    OpBitcast = 124,

    // Composite
    OpVectorExtractDynamic = 77,
    OpVectorInsertDynamic = 78,
    OpVectorShuffle = 79,
    OpCompositeConstruct = 80,
    OpCompositeExtract = 81,
    OpCompositeInsert = 82,
    OpCopyObject = 83,

    // Derivative
    OpDPdx = 207,
    OpDPdy = 208,
    OpFwidth = 209,

    // Barrier/Atomics
    OpControlBarrier = 224,
    OpMemoryBarrier = 225,
    OpAtomicLoad = 227,
    OpAtomicStore = 228,
    OpAtomicExchange = 229,
    OpAtomicCompareExchange = 230,
    OpAtomicIIncrement = 232,
    OpAtomicIDecrement = 233,
    OpAtomicIAdd = 234,
    OpAtomicISub = 235,
    OpAtomicSMin = 236,
    OpAtomicUMin = 237,
    OpAtomicSMax = 238,
    OpAtomicUMax = 239,
    OpAtomicAnd = 240,
    OpAtomicOr = 241,
    OpAtomicXor = 242,

    // Decoration
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpDecorationGroup = 73,
    OpGroupDecorate = 74,
    OpGroupMemberDecorate = 75,

    // Misc
    OpPhi = 245,
};

/// SPIR-V capabilities.
pub const Capability = enum(u32) {
    Matrix = 0,
    Shader = 1,
    Geometry = 2,
    Tessellation = 3,
    Addresses = 4,
    Linkage = 5,
    Kernel = 6,
    Vector16 = 7,
    Float16Buffer = 8,
    Float16 = 9,
    Float64 = 10,
    Int64 = 11,
    Int64Atomics = 12,
    ImageBasic = 13,
    ImageReadWrite = 14,
    ImageMipmap = 15,
    Pipes = 17,
    Groups = 18,
    DeviceEnqueue = 19,
    LiteralSampler = 20,
    AtomicStorage = 21,
    Int16 = 22,
    TessellationPointSize = 23,
    GeometryPointSize = 24,
    ImageGatherExtended = 25,
    StorageImageMultisample = 27,
    UniformBufferArrayDynamicIndexing = 28,
    SampledImageArrayDynamicIndexing = 29,
    StorageBufferArrayDynamicIndexing = 30,
    StorageImageArrayDynamicIndexing = 31,
    ClipDistance = 32,
    CullDistance = 33,
    ImageCubeArray = 34,
    SampleRateShading = 35,
    ImageRect = 36,
    SampledRect = 37,
    GenericPointer = 38,
    Int8 = 39,
    InputAttachment = 40,
    SparseResidency = 41,
    MinLod = 42,
    Sampled1D = 43,
    Image1D = 44,
    SampledCubeArray = 45,
    SampledBuffer = 46,
    ImageBuffer = 47,
    ImageMSArray = 48,
    StorageImageExtendedFormats = 49,
    ImageQuery = 50,
    DerivativeControl = 51,
    InterpolationFunction = 52,
    TransformFeedback = 53,
    GeometryStreams = 54,
    StorageImageReadWithoutFormat = 55,
    StorageImageWriteWithoutFormat = 56,
    MultiViewport = 57,
    SubgroupDispatch = 58,
    NamedBarrier = 59,
    PipeStorage = 60,
    GroupNonUniform = 61,
    GroupNonUniformVote = 62,
    GroupNonUniformArithmetic = 63,
    GroupNonUniformBallot = 64,
    GroupNonUniformShuffle = 65,
    GroupNonUniformShuffleRelative = 66,
    GroupNonUniformClustered = 67,
    GroupNonUniformQuad = 68,
    ShaderLayer = 69,
    ShaderViewportIndex = 70,
};

/// SPIR-V execution model.
pub const ExecutionModel = enum(u32) {
    Vertex = 0,
    TessellationControl = 1,
    TessellationEvaluation = 2,
    Geometry = 3,
    Fragment = 4,
    GLCompute = 5,
    Kernel = 6,
};

/// SPIR-V addressing model.
pub const AddressingModel = enum(u32) {
    Logical = 0,
    Physical32 = 1,
    Physical64 = 2,
    PhysicalStorageBuffer64 = 5348,
};

/// SPIR-V memory model.
pub const MemoryModel = enum(u32) {
    Simple = 0,
    GLSL450 = 1,
    OpenCL = 2,
    Vulkan = 3,
};

/// SPIR-V execution mode.
pub const ExecutionMode = enum(u32) {
    Invocations = 0,
    SpacingEqual = 1,
    SpacingFractionalEven = 2,
    SpacingFractionalOdd = 3,
    VertexOrderCw = 4,
    VertexOrderCcw = 5,
    PixelCenterInteger = 6,
    OriginUpperLeft = 7,
    OriginLowerLeft = 8,
    EarlyFragmentTests = 9,
    PointMode = 10,
    Xfb = 11,
    DepthReplacing = 12,
    DepthGreater = 14,
    DepthLess = 15,
    DepthUnchanged = 16,
    LocalSize = 17,
    LocalSizeHint = 18,
    InputPoints = 19,
    InputLines = 20,
    InputLinesAdjacency = 21,
    Triangles = 22,
    InputTrianglesAdjacency = 23,
    Quads = 24,
    Isolines = 25,
    OutputVertices = 26,
    OutputPoints = 27,
    OutputLineStrip = 28,
    OutputTriangleStrip = 29,
    VecTypeHint = 30,
    ContractionOff = 31,
};

/// SPIR-V storage class.
pub const StorageClass = enum(u32) {
    UniformConstant = 0,
    Input = 1,
    Uniform = 2,
    Output = 3,
    Workgroup = 4,
    CrossWorkgroup = 5,
    Private = 6,
    Function = 7,
    Generic = 8,
    PushConstant = 9,
    AtomicCounter = 10,
    Image = 11,
    StorageBuffer = 12,
};

/// SPIR-V decoration.
pub const Decoration = enum(u32) {
    RelaxedPrecision = 0,
    SpecId = 1,
    Block = 2,
    BufferBlock = 3,
    RowMajor = 4,
    ColMajor = 5,
    ArrayStride = 6,
    MatrixStride = 7,
    GLSLShared = 8,
    GLSLPacked = 9,
    CPacked = 10,
    BuiltIn = 11,
    NoPerspective = 13,
    Flat = 14,
    Patch = 15,
    Centroid = 16,
    Sample = 17,
    Invariant = 18,
    Restrict = 19,
    Aliased = 20,
    Volatile = 21,
    Constant = 22,
    Coherent = 23,
    NonWritable = 24,
    NonReadable = 25,
    Uniform = 26,
    UniformId = 27,
    SaturatedConversion = 28,
    Stream = 29,
    Location = 30,
    Component = 31,
    Index = 32,
    Binding = 33,
    DescriptorSet = 34,
    Offset = 35,
    XfbBuffer = 36,
    XfbStride = 37,
    FuncParamAttr = 38,
    FPRoundingMode = 39,
    FPFastMathMode = 40,
    LinkageAttributes = 41,
    NoContraction = 42,
    InputAttachmentIndex = 43,
    Alignment = 44,
    MaxByteOffset = 45,
    AlignmentId = 46,
    MaxByteOffsetId = 47,
};

/// SPIR-V built-in variables.
pub const BuiltIn = enum(u32) {
    Position = 0,
    PointSize = 1,
    ClipDistance = 3,
    CullDistance = 4,
    VertexId = 5,
    InstanceId = 6,
    PrimitiveId = 7,
    InvocationId = 8,
    Layer = 9,
    ViewportIndex = 10,
    TessLevelOuter = 11,
    TessLevelInner = 12,
    TessCoord = 13,
    PatchVertices = 14,
    FragCoord = 15,
    PointCoord = 16,
    FrontFacing = 17,
    SampleId = 18,
    SamplePosition = 19,
    SampleMask = 20,
    FragDepth = 22,
    HelperInvocation = 23,
    NumWorkgroups = 24,
    WorkgroupSize = 25,
    WorkgroupId = 26,
    LocalInvocationId = 27,
    GlobalInvocationId = 28,
    LocalInvocationIndex = 29,
    WorkDim = 30,
    GlobalSize = 31,
    EnqueuedWorkgroupSize = 32,
    GlobalOffset = 33,
    GlobalLinearId = 34,
    SubgroupSize = 36,
    SubgroupMaxSize = 37,
    NumSubgroups = 38,
    NumEnqueuedSubgroups = 39,
    SubgroupId = 40,
    SubgroupLocalInvocationId = 41,
    VertexIndex = 42,
    InstanceIndex = 43,
};

/// Memory semantics for barriers and atomics.
pub const MemorySemantics = enum(u32) {
    None = 0x0,
    Acquire = 0x2,
    Release = 0x4,
    AcquireRelease = 0x8,
    SequentiallyConsistent = 0x10,
    UniformMemory = 0x40,
    SubgroupMemory = 0x80,
    WorkgroupMemory = 0x100,
    CrossWorkgroupMemory = 0x200,
    AtomicCounterMemory = 0x400,
    ImageMemory = 0x800,
};

/// Scope for barriers.
pub const Scope = enum(u32) {
    CrossDevice = 0,
    Device = 1,
    Workgroup = 2,
    Subgroup = 3,
    Invocation = 4,
};

/// SPIR-V code generator.
pub const SpirvGenerator = struct {
    allocator: std.mem.Allocator,
    /// SPIR-V words (binary output).
    words: std.ArrayListUnmanaged(u32),
    /// Current ID bound.
    next_id: u32,
    /// Type ID cache.
    type_ids: std.AutoHashMapUnmanaged(TypeKey, u32),
    /// Constant ID cache.
    const_ids: std.AutoHashMapUnmanaged(ConstKey, u32),
    /// Variable ID map (name -> id).
    var_ids: std.StringHashMapUnmanaged(u32),
    /// Type declarations (stored for later emission).
    type_section: std.ArrayListUnmanaged(u32),
    /// Constants section.
    const_section: std.ArrayListUnmanaged(u32),
    /// Annotations section.
    annotation_section: std.ArrayListUnmanaged(u32),
    /// Debug names section.
    debug_section: std.ArrayListUnmanaged(u32),
    /// Entry points section.
    entry_section: std.ArrayListUnmanaged(u32),
    /// Function section.
    function_section: std.ArrayListUnmanaged(u32),
    /// GLSL.std.450 extended instruction set ID.
    glsl_ext_id: u32,

    const Self = @This();

    /// Key for type caching.
    const TypeKey = struct {
        tag: enum { void_, bool_, int, float, vector, matrix, array, runtime_array, ptr, struct_, function },
        data: u64,
        extra: u32,
    };

    /// Key for constant caching.
    const ConstKey = struct {
        type_id: u32,
        value: u64,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .words = .{},
            .next_id = 1,
            .type_ids = .{},
            .const_ids = .{},
            .var_ids = .{},
            .type_section = .{},
            .const_section = .{},
            .annotation_section = .{},
            .debug_section = .{},
            .entry_section = .{},
            .function_section = .{},
            .glsl_ext_id = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.words.deinit(self.allocator);
        self.type_ids.deinit(self.allocator);
        self.const_ids.deinit(self.allocator);
        self.var_ids.deinit(self.allocator);
        self.type_section.deinit(self.allocator);
        self.const_section.deinit(self.allocator);
        self.annotation_section.deinit(self.allocator);
        self.debug_section.deinit(self.allocator);
        self.entry_section.deinit(self.allocator);
        self.function_section.deinit(self.allocator);
    }

    /// Allocate a new ID.
    fn allocId(self: *Self) u32 {
        const id = self.next_id;
        self.next_id += 1;
        return id;
    }

    /// Generate SPIR-V binary from kernel IR.
    pub fn generate(
        self: *Self,
        ir: *const kernel.KernelIR,
    ) backend.CodegenError!backend.GeneratedSource {
        // Reset state
        self.next_id = 1;
        self.words.clearRetainingCapacity();
        self.type_ids.clearRetainingCapacity();
        self.const_ids.clearRetainingCapacity();
        self.var_ids.clearRetainingCapacity();
        self.type_section.clearRetainingCapacity();
        self.const_section.clearRetainingCapacity();
        self.annotation_section.clearRetainingCapacity();
        self.debug_section.clearRetainingCapacity();
        self.entry_section.clearRetainingCapacity();
        self.function_section.clearRetainingCapacity();

        // Reserve header space (5 words)
        try self.words.appendSlice(self.allocator, &[_]u32{ 0, 0, 0, 0, 0 });

        // Import GLSL.std.450 extended instruction set
        self.glsl_ext_id = self.allocId();
        try self.emitExtInstImport(self.glsl_ext_id, "GLSL.std.450");

        // Emit capabilities
        try self.emitCapability(.Shader);
        if (ir.required_features.fp64) {
            try self.emitCapability(.Float64);
        }
        if (ir.required_features.int64) {
            try self.emitCapability(.Int64);
        }
        if (ir.required_features.subgroups) {
            try self.emitCapability(.GroupNonUniform);
        }

        // Memory model
        try self.emitMemoryModel(.Logical, .GLSL450);

        // Generate types needed for built-ins and buffers
        const void_type = try self.getVoidType();
        const func_type = try self.getFunctionType(void_type, &.{});

        // Generate built-in variables
        const uvec3_type = try self.getVectorType(try self.getIntType(32, false), 3);
        const uint_type = try self.getIntType(32, false);
        const uvec3_ptr_input = try self.getPointerType(uvec3_type, .Input);
        const uint_ptr_input = try self.getPointerType(uint_type, .Input);

        // Create built-in variables
        const global_inv_id = self.allocId();
        const local_inv_id = self.allocId();
        const workgroup_id_var = self.allocId();
        const local_inv_index = self.allocId();

        try self.emitVariable(&self.type_section, global_inv_id, uvec3_ptr_input, .Input, null);
        try self.emitVariable(&self.type_section, local_inv_id, uvec3_ptr_input, .Input, null);
        try self.emitVariable(&self.type_section, workgroup_id_var, uvec3_ptr_input, .Input, null);
        try self.emitVariable(&self.type_section, local_inv_index, uint_ptr_input, .Input, null);

        // Decorate built-ins
        try self.emitDecorate(&self.annotation_section, global_inv_id, .BuiltIn, &.{@intFromEnum(BuiltIn.GlobalInvocationId)});
        try self.emitDecorate(&self.annotation_section, local_inv_id, .BuiltIn, &.{@intFromEnum(BuiltIn.LocalInvocationId)});
        try self.emitDecorate(&self.annotation_section, workgroup_id_var, .BuiltIn, &.{@intFromEnum(BuiltIn.WorkgroupId)});
        try self.emitDecorate(&self.annotation_section, local_inv_index, .BuiltIn, &.{@intFromEnum(BuiltIn.LocalInvocationIndex)});

        // Register built-in variable names
        try self.var_ids.put(self.allocator, "globalInvocationId", global_inv_id);
        try self.var_ids.put(self.allocator, "localInvocationId", local_inv_id);
        try self.var_ids.put(self.allocator, "workgroupId", workgroup_id_var);
        try self.var_ids.put(self.allocator, "localInvocationIndex", local_inv_index);

        // Debug names
        try self.emitName(&self.debug_section, global_inv_id, "globalInvocationId");
        try self.emitName(&self.debug_section, local_inv_id, "localInvocationId");
        try self.emitName(&self.debug_section, workgroup_id_var, "workgroupId");
        try self.emitName(&self.debug_section, local_inv_index, "localInvocationIndex");

        // Generate buffer variables
        var interface_ids = std.ArrayListUnmanaged(u32){};
        defer interface_ids.deinit(self.allocator);

        // Add built-ins to interface
        try interface_ids.append(self.allocator, global_inv_id);
        try interface_ids.append(self.allocator, local_inv_id);
        try interface_ids.append(self.allocator, workgroup_id_var);
        try interface_ids.append(self.allocator, local_inv_index);

        // Process buffers
        for (ir.buffers) |buf| {
            const elem_type = try self.typeFromIR(buf.element_type);
            const runtime_arr = try self.getRuntimeArrayType(elem_type);

            // Create block struct
            const struct_id = self.allocId();
            try self.emitTypeStruct(&self.type_section, struct_id, &.{runtime_arr});

            // Decorate struct and member
            try self.emitDecorate(&self.annotation_section, struct_id, .Block, &.{});
            try self.emitMemberDecorate(&self.annotation_section, struct_id, 0, .Offset, &.{0});

            // Array stride
            const elem_size = self.getTypeSize(buf.element_type);
            try self.emitDecorate(&self.annotation_section, runtime_arr, .ArrayStride, &.{elem_size});

            // Create pointer and variable
            const ptr_type = try self.getPointerType(struct_id, .StorageBuffer);
            const var_id = self.allocId();
            try self.emitVariable(&self.type_section, var_id, ptr_type, .StorageBuffer, null);

            // Decorate binding
            try self.emitDecorate(&self.annotation_section, var_id, .DescriptorSet, &.{buf.group});
            try self.emitDecorate(&self.annotation_section, var_id, .Binding, &.{buf.binding});

            // Decorate access
            if (buf.access == .read_only) {
                try self.emitDecorate(&self.annotation_section, var_id, .NonWritable, &.{});
            }

            try self.var_ids.put(self.allocator, buf.name, var_id);
            try self.emitName(&self.debug_section, var_id, buf.name);
            try interface_ids.append(self.allocator, var_id);
        }

        // Process uniforms
        if (ir.uniforms.len > 0) {
            // Create uniform block struct
            var member_types = std.ArrayListUnmanaged(u32){};
            defer member_types.deinit(self.allocator);

            var member_offset: u32 = 0;
            const struct_id = self.allocId();

            for (ir.uniforms, 0..) |uni, i| {
                const member_type = try self.typeFromIR(uni.ty);
                try member_types.append(self.allocator, member_type);

                // Decorate member offset
                try self.emitMemberDecorate(&self.annotation_section, struct_id, @intCast(i), .Offset, &.{member_offset});
                try self.emitMemberName(&self.debug_section, struct_id, @intCast(i), uni.name);

                member_offset += self.getTypeSize(uni.ty);
            }

            try self.emitTypeStruct(&self.type_section, struct_id, member_types.items);
            try self.emitDecorate(&self.annotation_section, struct_id, .Block, &.{});

            const ptr_type = try self.getPointerType(struct_id, .Uniform);
            const var_id = self.allocId();
            try self.emitVariable(&self.type_section, var_id, ptr_type, .Uniform, null);
            try self.emitDecorate(&self.annotation_section, var_id, .DescriptorSet, &.{0});
            try self.emitDecorate(&self.annotation_section, var_id, .Binding, &.{0});

            try self.var_ids.put(self.allocator, "uniforms", var_id);
            try self.emitName(&self.debug_section, var_id, "uniforms");
            try interface_ids.append(self.allocator, var_id);
        }

        // Process shared memory
        for (ir.shared_memory) |shared| {
            const elem_type = try self.typeFromIR(shared.element_type);
            const arr_type = if (shared.size) |size|
                try self.getArrayType(elem_type, @intCast(size))
            else
                try self.getRuntimeArrayType(elem_type);

            const ptr_type = try self.getPointerType(arr_type, .Workgroup);
            const var_id = self.allocId();
            try self.emitVariable(&self.type_section, var_id, ptr_type, .Workgroup, null);

            try self.var_ids.put(self.allocator, shared.name, var_id);
            try self.emitName(&self.debug_section, var_id, shared.name);
        }

        // Entry point
        const main_func_id = self.allocId();
        try self.emitEntryPoint(.GLCompute, main_func_id, "main", interface_ids.items);
        try self.emitExecutionMode(main_func_id, .LocalSize, &.{
            ir.workgroup_size[0],
            ir.workgroup_size[1],
            ir.workgroup_size[2],
        });

        // Generate main function
        try self.emitFunction(&self.function_section, main_func_id, void_type, .{ .none = {} }, func_type);

        const entry_label = self.allocId();
        try self.emitLabel(&self.function_section, entry_label);

        // Generate function body
        for (ir.body) |s| {
            try self.generateStmt(s);
        }

        // Return and end function
        try self.emitReturn(&self.function_section);
        try self.emitFunctionEnd(&self.function_section);

        // Assemble final module
        try self.assembleModule();

        // Convert to byte slice
        const word_bytes = std.mem.sliceAsBytes(self.words.items);
        const code = try self.allocator.dupe(u8, word_bytes);
        const entry_point_name = try self.allocator.dupe(u8, "main");

        return .{
            .code = code,
            .entry_point = entry_point_name,
            .backend = .vulkan,
            .language = .spirv,
            .spirv_binary = code,
        };
    }

    /// Assemble the final SPIR-V module.
    fn assembleModule(self: *Self) !void {
        // Write header
        self.words.items[0] = SPIRV_MAGIC;
        self.words.items[1] = SPIRV_VERSION;
        self.words.items[2] = GENERATOR_ID;
        self.words.items[3] = self.next_id; // Bound
        self.words.items[4] = 0; // Schema

        // Sections are already in words from capabilities/memory model
        // Now append other sections in order
        try self.words.appendSlice(self.allocator, self.entry_section.items);
        try self.words.appendSlice(self.allocator, self.debug_section.items);
        try self.words.appendSlice(self.allocator, self.annotation_section.items);
        try self.words.appendSlice(self.allocator, self.type_section.items);
        try self.words.appendSlice(self.allocator, self.const_section.items);
        try self.words.appendSlice(self.allocator, self.function_section.items);
    }

    /// Generate code for a statement.
    fn generateStmt(self: *Self, s: *const stmt.Stmt) !void {
        switch (s.*) {
            .var_decl => |v| {
                const ty = try self.typeFromIR(v.ty);
                const ptr_type = try self.getPointerType(ty, .Function);
                const var_id = self.allocId();

                // Initialize variable
                const init_id: ?u32 = if (v.init) |init_expr|
                    try self.generateExpr(init_expr)
                else
                    null;

                try self.emitVariable(&self.function_section, var_id, ptr_type, .Function, init_id);
                try self.var_ids.put(self.allocator, v.name, var_id);
            },
            .assign => |a| {
                const value_id = try self.generateExpr(a.value);
                const target_id = try self.generateExprPtr(a.target);
                try self.emitStore(&self.function_section, target_id, value_id);
            },
            .compound_assign => |ca| {
                const target_ptr = try self.generateExprPtr(ca.target);
                const current_val = try self.generateLoad(target_ptr);
                const operand_val = try self.generateExpr(ca.value);
                const result = try self.generateBinaryOp(ca.op, current_val, operand_val);
                try self.emitStore(&self.function_section, target_ptr, result);
            },
            .if_ => |i| {
                const cond_id = try self.generateExpr(i.condition);
                const then_label = self.allocId();
                const else_label = self.allocId();
                const merge_label = self.allocId();

                try self.emitSelectionMerge(&self.function_section, merge_label);
                try self.emitBranchConditional(&self.function_section, cond_id, then_label, if (i.else_body != null) else_label else merge_label);

                // Then block
                try self.emitLabel(&self.function_section, then_label);
                for (i.then_body) |body_stmt| {
                    try self.generateStmt(body_stmt);
                }
                try self.emitBranch(&self.function_section, merge_label);

                // Else block
                if (i.else_body) |else_body| {
                    try self.emitLabel(&self.function_section, else_label);
                    for (else_body) |body_stmt| {
                        try self.generateStmt(body_stmt);
                    }
                    try self.emitBranch(&self.function_section, merge_label);
                }

                // Merge block
                try self.emitLabel(&self.function_section, merge_label);
            },
            .for_ => |f| {
                // Initialize
                if (f.init) |init_stmt| {
                    try self.generateStmt(init_stmt);
                }

                const header_label = self.allocId();
                const body_label = self.allocId();
                const continue_label = self.allocId();
                const merge_label = self.allocId();

                try self.emitBranch(&self.function_section, header_label);
                try self.emitLabel(&self.function_section, header_label);
                try self.emitLoopMerge(&self.function_section, merge_label, continue_label);

                if (f.condition) |cond| {
                    const cond_id = try self.generateExpr(cond);
                    try self.emitBranchConditional(&self.function_section, cond_id, body_label, merge_label);
                } else {
                    try self.emitBranch(&self.function_section, body_label);
                }

                // Body
                try self.emitLabel(&self.function_section, body_label);
                for (f.body) |body_stmt| {
                    try self.generateStmt(body_stmt);
                }
                try self.emitBranch(&self.function_section, continue_label);

                // Continue block
                try self.emitLabel(&self.function_section, continue_label);
                if (f.update) |update| {
                    try self.generateStmt(update);
                }
                try self.emitBranch(&self.function_section, header_label);

                // Merge
                try self.emitLabel(&self.function_section, merge_label);
            },
            .while_ => |w| {
                const header_label = self.allocId();
                const body_label = self.allocId();
                const merge_label = self.allocId();

                try self.emitBranch(&self.function_section, header_label);
                try self.emitLabel(&self.function_section, header_label);
                try self.emitLoopMerge(&self.function_section, merge_label, header_label);

                const cond_id = try self.generateExpr(w.condition);
                try self.emitBranchConditional(&self.function_section, cond_id, body_label, merge_label);

                try self.emitLabel(&self.function_section, body_label);
                for (w.body) |body_stmt| {
                    try self.generateStmt(body_stmt);
                }
                try self.emitBranch(&self.function_section, header_label);

                try self.emitLabel(&self.function_section, merge_label);
            },
            .return_ => {
                try self.emitReturn(&self.function_section);
            },
            .break_ => {
                // Would need to track current loop merge label
            },
            .continue_ => {
                // Would need to track current loop continue label
            },
            .expr_stmt => |e| {
                _ = try self.generateExpr(e);
            },
            .block => |b| {
                for (b.statements) |body_stmt| {
                    try self.generateStmt(body_stmt);
                }
            },
            else => {},
        }
    }

    /// Generate an expression and return its result ID.
    fn generateExpr(self: *Self, e: *const expr.Expr) !u32 {
        return switch (e.*) {
            .literal => |lit| try self.generateLiteral(lit),
            .ref => |ref| {
                if (ref.name) |name| {
                    if (self.var_ids.get(name)) |var_id| {
                        return try self.generateLoad(var_id);
                    }
                }
                return error.InvalidIR;
            },
            .unary => |un| {
                const operand_id = try self.generateExpr(un.operand);
                return try self.generateUnaryOp(un.op, operand_id);
            },
            .binary => |bin| {
                const left_id = try self.generateExpr(bin.left);
                const right_id = try self.generateExpr(bin.right);
                return try self.generateBinaryOp(bin.op, left_id, right_id);
            },
            .call => |c| try self.generateCall(c),
            .index => |idx| {
                const base_ptr = try self.generateExprPtr(idx.base);
                const index_id = try self.generateExpr(idx.index);
                const elem_ptr = try self.generateAccessChain(base_ptr, index_id);
                return try self.generateLoad(elem_ptr);
            },
            .field => |f| {
                const base_ptr = try self.generateExprPtr(f.base);
                // Field index would need type info
                const zero = try self.getConstantU32(0);
                const field_ptr = try self.generateAccessChain(base_ptr, zero);
                return try self.generateLoad(field_ptr);
            },
            .cast => |c| {
                const operand_id = try self.generateExpr(c.operand);
                const target_type = try self.typeFromIR(c.target_type);
                return try self.generateCast(operand_id, target_type);
            },
            .select => |s| {
                const cond_id = try self.generateExpr(s.condition);
                const true_id = try self.generateExpr(s.true_value);
                const false_id = try self.generateExpr(s.false_value);
                const result_id = self.allocId();
                const float_type = try self.getFloatType(32);
                try self.emitOp(&self.function_section, .OpSelect, &.{ float_type, result_id, cond_id, true_id, false_id });
                return result_id;
            },
            .vector_construct => |vc| {
                var component_ids = std.ArrayListUnmanaged(u32){};
                defer component_ids.deinit(self.allocator);

                for (vc.components) |comp| {
                    try component_ids.append(self.allocator, try self.generateExpr(comp));
                }

                const elem_type = try self.scalarTypeFromIR(vc.element_type);
                const vec_type = try self.getVectorType(elem_type, vc.size);
                const result_id = self.allocId();

                var operands = std.ArrayListUnmanaged(u32){};
                defer operands.deinit(self.allocator);
                try operands.append(self.allocator, vec_type);
                try operands.append(self.allocator, result_id);
                try operands.appendSlice(self.allocator, component_ids.items);

                try self.emitOp(&self.function_section, .OpCompositeConstruct, operands.items);
                return result_id;
            },
            .swizzle => |sw| {
                const base_id = try self.generateExpr(sw.base);
                if (sw.components.len == 1) {
                    // Extract single component
                    const result_id = self.allocId();
                    const float_type = try self.getFloatType(32);
                    try self.emitOp(&self.function_section, .OpCompositeExtract, &.{ float_type, result_id, base_id, sw.components[0] });
                    return result_id;
                } else {
                    // Vector shuffle
                    const elem_type = try self.getFloatType(32);
                    const vec_type = try self.getVectorType(elem_type, @intCast(sw.components.len));
                    const result_id = self.allocId();

                    var operands = std.ArrayListUnmanaged(u32){};
                    defer operands.deinit(self.allocator);
                    try operands.append(self.allocator, vec_type);
                    try operands.append(self.allocator, result_id);
                    try operands.append(self.allocator, base_id);
                    try operands.append(self.allocator, base_id);
                    for (sw.components) |c| {
                        try operands.append(self.allocator, c);
                    }

                    try self.emitOp(&self.function_section, .OpVectorShuffle, operands.items);
                    return result_id;
                }
            },
        };
    }

    /// Generate expression as pointer (for assignments).
    fn generateExprPtr(self: *Self, e: *const expr.Expr) !u32 {
        return switch (e.*) {
            .ref => |ref| {
                if (ref.name) |name| {
                    if (self.var_ids.get(name)) |var_id| {
                        return var_id;
                    }
                }
                return error.InvalidIR;
            },
            .index => |idx| {
                const base_ptr = try self.generateExprPtr(idx.base);
                const index_id = try self.generateExpr(idx.index);
                return try self.generateAccessChain(base_ptr, index_id);
            },
            .field => |f| {
                const base_ptr = try self.generateExprPtr(f.base);
                const zero = try self.getConstantU32(0);
                return try self.generateAccessChain(base_ptr, zero);
            },
            else => error.InvalidIR,
        };
    }

    /// Generate a literal constant.
    fn generateLiteral(self: *Self, lit: expr.Literal) !u32 {
        return switch (lit) {
            .bool_ => |v| if (v) try self.getConstantTrue() else try self.getConstantFalse(),
            .i32_ => |v| try self.getConstantI32(v),
            .i64_ => |v| try self.getConstantI64(v),
            .u32_ => |v| try self.getConstantU32(v),
            .u64_ => |v| try self.getConstantU64(v),
            .f32_ => |v| try self.getConstantF32(v),
            .f64_ => |v| try self.getConstantF64(v),
        };
    }

    /// Generate a unary operation.
    fn generateUnaryOp(self: *Self, op: expr.UnaryOp, operand: u32) !u32 {
        const result_id = self.allocId();
        const float_type = try self.getFloatType(32);

        switch (op) {
            .neg => {
                try self.emitOp(&self.function_section, .OpFNegate, &.{ float_type, result_id, operand });
            },
            .not => {
                const bool_type = try self.getBoolType();
                try self.emitOp(&self.function_section, .OpLogicalNot, &.{ bool_type, result_id, operand });
            },
            .bit_not => {
                const int_type = try self.getIntType(32, true);
                try self.emitOp(&self.function_section, .OpNot, &.{ int_type, result_id, operand });
            },
            .sqrt, .sin, .cos, .tan, .exp, .log, .abs, .floor, .ceil, .round => {
                const ext_inst = switch (op) {
                    .sqrt => 31, // Sqrt
                    .sin => 13, // Sin
                    .cos => 14, // Cos
                    .tan => 15, // Tan
                    .exp => 27, // Exp
                    .log => 28, // Log
                    .abs => 4, // FAbs
                    .floor => 8, // Floor
                    .ceil => 9, // Ceil
                    .round => 1, // Round
                    else => 0,
                };
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, ext_inst, operand });
            },
            else => {
                // Fallback
                try self.emitOp(&self.function_section, .OpCopyObject, &.{ float_type, result_id, operand });
            },
        }
        return result_id;
    }

    /// Generate a binary operation.
    fn generateBinaryOp(self: *Self, op: expr.BinaryOp, left: u32, right: u32) !u32 {
        const result_id = self.allocId();
        const float_type = try self.getFloatType(32);
        const bool_type = try self.getBoolType();

        const opcode: OpCode = switch (op) {
            .add => .OpFAdd,
            .sub => .OpFSub,
            .mul => .OpFMul,
            .div => .OpFDiv,
            .mod => .OpFMod,
            .eq => .OpFOrdEqual,
            .ne => .OpFOrdNotEqual,
            .lt => .OpFOrdLessThan,
            .le => .OpFOrdLessThanEqual,
            .gt => .OpFOrdGreaterThan,
            .ge => .OpFOrdGreaterThanEqual,
            .and_ => .OpLogicalAnd,
            .or_ => .OpLogicalOr,
            .bit_and => .OpBitwiseAnd,
            .bit_or => .OpBitwiseOr,
            .bit_xor => .OpBitwiseXor,
            .shl => .OpShiftLeftLogical,
            .shr => .OpShiftRightLogical,
            else => .OpFAdd,
        };

        const result_type = if (op.isComparison()) bool_type else float_type;
        try self.emitOp(&self.function_section, opcode, &.{ result_type, result_id, left, right });
        return result_id;
    }

    /// Generate a function call.
    fn generateCall(self: *Self, c: expr.Expr.CallExpr) !u32 {
        switch (c.function) {
            .barrier => {
                const exec_scope = try self.getConstantU32(@intFromEnum(Scope.Workgroup));
                const mem_scope = try self.getConstantU32(@intFromEnum(Scope.Workgroup));
                const semantics = try self.getConstantU32(@intFromEnum(MemorySemantics.WorkgroupMemory) | @intFromEnum(MemorySemantics.AcquireRelease));
                try self.emitOp(&self.function_section, .OpControlBarrier, &.{ exec_scope, mem_scope, semantics });
                return 0;
            },
            .memory_barrier => {
                const scope = try self.getConstantU32(@intFromEnum(Scope.Device));
                const semantics = try self.getConstantU32(@intFromEnum(MemorySemantics.AcquireRelease));
                try self.emitOp(&self.function_section, .OpMemoryBarrier, &.{ scope, semantics });
                return 0;
            },
            .atomic_add => {
                if (c.args.len >= 2) {
                    const ptr = try self.generateExprPtr(c.args[0]);
                    const value = try self.generateExpr(c.args[1]);
                    const uint_type = try self.getIntType(32, false);
                    const scope = try self.getConstantU32(@intFromEnum(Scope.Device));
                    const semantics = try self.getConstantU32(0);
                    const result_id = self.allocId();
                    try self.emitOp(&self.function_section, .OpAtomicIAdd, &.{ uint_type, result_id, ptr, scope, semantics, value });
                    return result_id;
                }
                return 0;
            },
            .clamp => {
                if (c.args.len >= 3) {
                    const x = try self.generateExpr(c.args[0]);
                    const min_val = try self.generateExpr(c.args[1]);
                    const max_val = try self.generateExpr(c.args[2]);
                    const float_type = try self.getFloatType(32);
                    const result_id = self.allocId();
                    // GLSL.std.450 FClamp = 43
                    try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, 43, x, min_val, max_val });
                    return result_id;
                }
                return 0;
            },
            .mix => {
                if (c.args.len >= 3) {
                    const x = try self.generateExpr(c.args[0]);
                    const y = try self.generateExpr(c.args[1]);
                    const a = try self.generateExpr(c.args[2]);
                    const float_type = try self.getFloatType(32);
                    const result_id = self.allocId();
                    // GLSL.std.450 FMix = 46
                    try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, 46, x, y, a });
                    return result_id;
                }
                return 0;
            },
            .fma => {
                if (c.args.len >= 3) {
                    const a = try self.generateExpr(c.args[0]);
                    const b = try self.generateExpr(c.args[1]);
                    const cc = try self.generateExpr(c.args[2]);
                    const float_type = try self.getFloatType(32);
                    const result_id = self.allocId();
                    // GLSL.std.450 Fma = 50
                    try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, 50, a, b, cc });
                    return result_id;
                }
                return 0;
            },
            else => return 0,
        }
    }

    /// Generate load from pointer.
    fn generateLoad(self: *Self, ptr: u32) !u32 {
        const result_id = self.allocId();
        const float_type = try self.getFloatType(32);
        try self.emitOp(&self.function_section, .OpLoad, &.{ float_type, result_id, ptr });
        return result_id;
    }

    /// Generate access chain.
    fn generateAccessChain(self: *Self, base: u32, index: u32) !u32 {
        const result_id = self.allocId();
        const float_type = try self.getFloatType(32);
        const ptr_type = try self.getPointerType(float_type, .StorageBuffer);
        try self.emitOp(&self.function_section, .OpAccessChain, &.{ ptr_type, result_id, base, index });
        return result_id;
    }

    /// Generate type cast.
    fn generateCast(self: *Self, value: u32, target_type: u32) !u32 {
        const result_id = self.allocId();
        try self.emitOp(&self.function_section, .OpBitcast, &.{ target_type, result_id, value });
        return result_id;
    }

    // =========================================================================
    // Type Generation
    // =========================================================================

    fn typeFromIR(self: *Self, ty: types.Type) !u32 {
        return switch (ty) {
            .scalar => |s| try self.scalarTypeFromIR(s),
            .vector => |v| try self.getVectorType(try self.scalarTypeFromIR(v.element), v.size),
            .array => |a| {
                const elem = try self.typeFromIR(a.element.*);
                if (a.size) |size| {
                    return try self.getArrayType(elem, @intCast(size));
                } else {
                    return try self.getRuntimeArrayType(elem);
                }
            },
            .ptr => |p| {
                const pointee = try self.typeFromIR(p.pointee.*);
                const storage_class: StorageClass = switch (p.address_space) {
                    .private => .Private,
                    .workgroup => .Workgroup,
                    .storage => .StorageBuffer,
                    .uniform => .Uniform,
                };
                return try self.getPointerType(pointee, storage_class);
            },
            .void_ => try self.getVoidType(),
            .matrix => |m| {
                const vec = try self.getVectorType(try self.scalarTypeFromIR(m.element), m.rows);
                return try self.getMatrixType(vec, m.cols);
            },
        };
    }

    fn scalarTypeFromIR(self: *Self, s: types.ScalarType) !u32 {
        return switch (s) {
            .bool_ => try self.getBoolType(),
            .i8, .i16, .i32 => try self.getIntType(32, true),
            .i64 => try self.getIntType(64, true),
            .u8, .u16, .u32 => try self.getIntType(32, false),
            .u64 => try self.getIntType(64, false),
            .f16 => try self.getFloatType(16),
            .f32 => try self.getFloatType(32),
            .f64 => try self.getFloatType(64),
        };
    }

    fn getTypeSize(self: *Self, ty: types.Type) u32 {
        _ = self;
        return switch (ty) {
            .scalar => |s| @as(u32, s.byteSize()),
            .vector => |v| @as(u32, v.element.byteSize()) * @as(u32, v.size),
            else => 4,
        };
    }

    fn getVoidType(self: *Self) !u32 {
        const key = TypeKey{ .tag = .void_, .data = 0, .extra = 0 };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.type_section, .OpTypeVoid, &.{id});
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    fn getBoolType(self: *Self) !u32 {
        const key = TypeKey{ .tag = .bool_, .data = 0, .extra = 0 };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.type_section, .OpTypeBool, &.{id});
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    fn getIntType(self: *Self, width: u32, signed: bool) !u32 {
        const key = TypeKey{ .tag = .int, .data = width, .extra = if (signed) 1 else 0 };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.type_section, .OpTypeInt, &.{ id, width, if (signed) @as(u32, 1) else 0 });
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    fn getFloatType(self: *Self, width: u32) !u32 {
        const key = TypeKey{ .tag = .float, .data = width, .extra = 0 };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.type_section, .OpTypeFloat, &.{ id, width });
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    fn getVectorType(self: *Self, element_type: u32, count: u8) !u32 {
        const key = TypeKey{ .tag = .vector, .data = element_type, .extra = count };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.type_section, .OpTypeVector, &.{ id, element_type, count });
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    fn getMatrixType(self: *Self, column_type: u32, column_count: u8) !u32 {
        const key = TypeKey{ .tag = .matrix, .data = column_type, .extra = column_count };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.type_section, .OpTypeMatrix, &.{ id, column_type, column_count });
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    fn getArrayType(self: *Self, element_type: u32, length: u32) !u32 {
        const length_const = try self.getConstantU32(length);
        const key = TypeKey{ .tag = .array, .data = element_type, .extra = length };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.type_section, .OpTypeArray, &.{ id, element_type, length_const });
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    fn getRuntimeArrayType(self: *Self, element_type: u32) !u32 {
        const key = TypeKey{ .tag = .runtime_array, .data = element_type, .extra = 0 };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.type_section, .OpTypeRuntimeArray, &.{ id, element_type });
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    fn getPointerType(self: *Self, pointee_type: u32, storage_class: StorageClass) !u32 {
        const key = TypeKey{ .tag = .ptr, .data = pointee_type, .extra = @intFromEnum(storage_class) };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.type_section, .OpTypePointer, &.{ id, @intFromEnum(storage_class), pointee_type });
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    fn getFunctionType(self: *Self, return_type: u32, param_types: []const u32) !u32 {
        const key = TypeKey{ .tag = .function, .data = return_type, .extra = @intCast(param_types.len) };
        if (self.type_ids.get(key)) |id| return id;

        const id = self.allocId();
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, id);
        try operands.append(self.allocator, return_type);
        try operands.appendSlice(self.allocator, param_types);
        try self.emitOp(&self.type_section, .OpTypeFunction, operands.items);
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    // =========================================================================
    // Constant Generation
    // =========================================================================

    fn getConstantTrue(self: *Self) !u32 {
        const bool_type = try self.getBoolType();
        const key = ConstKey{ .type_id = bool_type, .value = 1 };
        if (self.const_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.const_section, .OpConstantTrue, &.{ bool_type, id });
        try self.const_ids.put(self.allocator, key, id);
        return id;
    }

    fn getConstantFalse(self: *Self) !u32 {
        const bool_type = try self.getBoolType();
        const key = ConstKey{ .type_id = bool_type, .value = 0 };
        if (self.const_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.const_section, .OpConstantFalse, &.{ bool_type, id });
        try self.const_ids.put(self.allocator, key, id);
        return id;
    }

    fn getConstantI32(self: *Self, value: i32) !u32 {
        const int_type = try self.getIntType(32, true);
        const key = ConstKey{ .type_id = int_type, .value = @bitCast(@as(u64, @bitCast(@as(i64, value)))) };
        if (self.const_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.const_section, .OpConstant, &.{ int_type, id, @as(u32, @bitCast(value)) });
        try self.const_ids.put(self.allocator, key, id);
        return id;
    }

    fn getConstantU32(self: *Self, value: u32) !u32 {
        const int_type = try self.getIntType(32, false);
        const key = ConstKey{ .type_id = int_type, .value = value };
        if (self.const_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.const_section, .OpConstant, &.{ int_type, id, value });
        try self.const_ids.put(self.allocator, key, id);
        return id;
    }

    fn getConstantI64(self: *Self, value: i64) !u32 {
        const int_type = try self.getIntType(64, true);
        const bits: u64 = @bitCast(value);
        const key = ConstKey{ .type_id = int_type, .value = bits };
        if (self.const_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.const_section, .OpConstant, &.{ int_type, id, @as(u32, @truncate(bits)), @as(u32, @truncate(bits >> 32)) });
        try self.const_ids.put(self.allocator, key, id);
        return id;
    }

    fn getConstantU64(self: *Self, value: u64) !u32 {
        const int_type = try self.getIntType(64, false);
        const key = ConstKey{ .type_id = int_type, .value = value };
        if (self.const_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.const_section, .OpConstant, &.{ int_type, id, @as(u32, @truncate(value)), @as(u32, @truncate(value >> 32)) });
        try self.const_ids.put(self.allocator, key, id);
        return id;
    }

    fn getConstantF32(self: *Self, value: f32) !u32 {
        const float_type = try self.getFloatType(32);
        const bits: u32 = @bitCast(value);
        const key = ConstKey{ .type_id = float_type, .value = bits };
        if (self.const_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.const_section, .OpConstant, &.{ float_type, id, bits });
        try self.const_ids.put(self.allocator, key, id);
        return id;
    }

    fn getConstantF64(self: *Self, value: f64) !u32 {
        const float_type = try self.getFloatType(64);
        const bits: u64 = @bitCast(value);
        const key = ConstKey{ .type_id = float_type, .value = bits };
        if (self.const_ids.get(key)) |id| return id;

        const id = self.allocId();
        try self.emitOp(&self.const_section, .OpConstant, &.{ float_type, id, @as(u32, @truncate(bits)), @as(u32, @truncate(bits >> 32)) });
        try self.const_ids.put(self.allocator, key, id);
        return id;
    }

    // =========================================================================
    // Instruction Emission
    // =========================================================================

    fn emitOp(self: *Self, section: *std.ArrayListUnmanaged(u32), opcode: OpCode, operands: []const u32) !void {
        const word_count: u32 = @intCast(1 + operands.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(opcode));
        try section.append(self.allocator, first_word);
        try section.appendSlice(self.allocator, operands);
    }

    fn emitCapability(self: *Self, cap: Capability) !void {
        try self.emitOp(&self.words, .OpCapability, &.{@intFromEnum(cap)});
    }

    fn emitExtInstImport(self: *Self, result_id: u32, name_str: []const u8) !void {
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, result_id);
        try self.appendString(&operands, name_str);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpExtInstImport));
        try self.words.append(self.allocator, first_word);
        try self.words.appendSlice(self.allocator, operands.items);
    }

    fn emitMemoryModel(self: *Self, addressing: AddressingModel, memory: MemoryModel) !void {
        try self.emitOp(&self.words, .OpMemoryModel, &.{ @intFromEnum(addressing), @intFromEnum(memory) });
    }

    fn emitEntryPoint(self: *Self, model: ExecutionModel, func_id: u32, name_str: []const u8, interface: []const u32) !void {
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, @intFromEnum(model));
        try operands.append(self.allocator, func_id);
        try self.appendString(&operands, name_str);
        try operands.appendSlice(self.allocator, interface);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpEntryPoint));
        try self.entry_section.append(self.allocator, first_word);
        try self.entry_section.appendSlice(self.allocator, operands.items);
    }

    fn emitExecutionMode(self: *Self, func_id: u32, mode: ExecutionMode, params: []const u32) !void {
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, func_id);
        try operands.append(self.allocator, @intFromEnum(mode));
        try operands.appendSlice(self.allocator, params);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpExecutionMode));
        try self.entry_section.append(self.allocator, first_word);
        try self.entry_section.appendSlice(self.allocator, operands.items);
    }

    fn emitName(self: *Self, section: *std.ArrayListUnmanaged(u32), id: u32, name_str: []const u8) !void {
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, id);
        try self.appendString(&operands, name_str);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpName));
        try section.append(self.allocator, first_word);
        try section.appendSlice(self.allocator, operands.items);
    }

    fn emitMemberName(self: *Self, section: *std.ArrayListUnmanaged(u32), type_id: u32, member: u32, name_str: []const u8) !void {
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, type_id);
        try operands.append(self.allocator, member);
        try self.appendString(&operands, name_str);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpMemberName));
        try section.append(self.allocator, first_word);
        try section.appendSlice(self.allocator, operands.items);
    }

    fn emitDecorate(self: *Self, section: *std.ArrayListUnmanaged(u32), target: u32, decoration: Decoration, params: []const u32) !void {
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, target);
        try operands.append(self.allocator, @intFromEnum(decoration));
        try operands.appendSlice(self.allocator, params);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpDecorate));
        try section.append(self.allocator, first_word);
        try section.appendSlice(self.allocator, operands.items);
    }

    fn emitMemberDecorate(self: *Self, section: *std.ArrayListUnmanaged(u32), struct_type: u32, member: u32, decoration: Decoration, params: []const u32) !void {
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, struct_type);
        try operands.append(self.allocator, member);
        try operands.append(self.allocator, @intFromEnum(decoration));
        try operands.appendSlice(self.allocator, params);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpMemberDecorate));
        try section.append(self.allocator, first_word);
        try section.appendSlice(self.allocator, operands.items);
    }

    fn emitTypeStruct(self: *Self, section: *std.ArrayListUnmanaged(u32), result_id: u32, member_types: []const u32) !void {
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, result_id);
        try operands.appendSlice(self.allocator, member_types);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpTypeStruct));
        try section.append(self.allocator, first_word);
        try section.appendSlice(self.allocator, operands.items);
    }

    fn emitVariable(self: *Self, section: *std.ArrayListUnmanaged(u32), result_id: u32, type_id: u32, storage_class: StorageClass, initializer: ?u32) !void {
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, type_id);
        try operands.append(self.allocator, result_id);
        try operands.append(self.allocator, @intFromEnum(storage_class));
        if (initializer) |init_id| {
            try operands.append(self.allocator, init_id);
        }

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpVariable));
        try section.append(self.allocator, first_word);
        try section.appendSlice(self.allocator, operands.items);
    }

    fn emitFunction(self: *Self, section: *std.ArrayListUnmanaged(u32), result_id: u32, return_type: u32, control: anytype, func_type: u32) !void {
        _ = control;
        try self.emitOp(section, .OpFunction, &.{ return_type, result_id, 0, func_type });
    }

    fn emitFunctionEnd(self: *Self, section: *std.ArrayListUnmanaged(u32)) !void {
        try self.emitOp(section, .OpFunctionEnd, &.{});
    }

    fn emitLabel(self: *Self, section: *std.ArrayListUnmanaged(u32), id: u32) !void {
        try self.emitOp(section, .OpLabel, &.{id});
    }

    fn emitReturn(self: *Self, section: *std.ArrayListUnmanaged(u32)) !void {
        try self.emitOp(section, .OpReturn, &.{});
    }

    fn emitBranch(self: *Self, section: *std.ArrayListUnmanaged(u32), target: u32) !void {
        try self.emitOp(section, .OpBranch, &.{target});
    }

    fn emitBranchConditional(self: *Self, section: *std.ArrayListUnmanaged(u32), condition: u32, true_label: u32, false_label: u32) !void {
        try self.emitOp(section, .OpBranchConditional, &.{ condition, true_label, false_label });
    }

    fn emitSelectionMerge(self: *Self, section: *std.ArrayListUnmanaged(u32), merge_label: u32) !void {
        try self.emitOp(section, .OpSelectionMerge, &.{ merge_label, 0 }); // 0 = None
    }

    fn emitLoopMerge(self: *Self, section: *std.ArrayListUnmanaged(u32), merge_label: u32, continue_label: u32) !void {
        try self.emitOp(section, .OpLoopMerge, &.{ merge_label, continue_label, 0 }); // 0 = None
    }

    fn emitStore(self: *Self, section: *std.ArrayListUnmanaged(u32), ptr: u32, value: u32) !void {
        try self.emitOp(section, .OpStore, &.{ ptr, value });
    }

    fn appendString(self: *Self, operands: *std.ArrayListUnmanaged(u32), str: []const u8) !void {
        var word: u32 = 0;
        var byte_idx: usize = 0;

        for (str) |c| {
            word |= @as(u32, c) << @intCast(byte_idx * 8);
            byte_idx += 1;
            if (byte_idx == 4) {
                try operands.append(self.allocator, word);
                word = 0;
                byte_idx = 0;
            }
        }
        // Append null terminator
        try operands.append(self.allocator, word); // Includes null terminator in remaining bytes
    }
};

// ============================================================================
// Tests
// ============================================================================

test "SpirvGenerator basic kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = SpirvGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    // Verify SPIR-V magic number
    const words = std.mem.bytesAsSlice(u32, result.code);
    try std.testing.expectEqual(SPIRV_MAGIC, words[0]);
    try std.testing.expectEqual(SPIRV_VERSION, words[1]);

    // Result should have spirv_binary set
    try std.testing.expect(result.spirv_binary != null);
}

test "SpirvGenerator with buffers" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = SpirvGenerator.init(allocator);
    defer gen.deinit();

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "input",
            .binding = 0,
            .group = 0,
            .element_type = types.Type.f32Type(),
            .access = .read_only,
        },
        .{
            .name = "output",
            .binding = 1,
            .group = 0,
            .element_type = types.Type.f32Type(),
            .access = .write_only,
        },
    };

    var ir = kernel.KernelIR.empty("vector_add");
    ir.buffers = &buffers;
    ir.workgroup_size = .{ 64, 1, 1 };

    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    // Verify valid SPIR-V
    const words = std.mem.bytesAsSlice(u32, result.code);
    try std.testing.expectEqual(SPIRV_MAGIC, words[0]);
    try std.testing.expect(words.len > 10); // Should have significant content
}

test "SpirvGenerator OpCode encoding" {
    // Verify instruction encoding
    const word_count: u32 = 4;
    const opcode: u32 = @intFromEnum(OpCode.OpCapability);
    const first_word = (word_count << 16) | opcode;

    try std.testing.expectEqual(@as(u32, 0x00040011), first_word);
}

test "SPIR-V module structure validation" {
    // This test validates the SPIR-V binary structure follows the spec
    // Reference: SPIR-V Specification Section 2.3 (Physical Layout of a SPIR-V Module)
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = SpirvGenerator.init(allocator);
    defer gen.deinit();

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "data",
            .binding = 0,
            .group = 0,
            .element_type = types.Type.f32Type(),
            .access = .read_write,
        },
    };

    var ir = kernel.KernelIR.empty("validation_test");
    ir.buffers = &buffers;
    ir.workgroup_size = .{ 256, 1, 1 };

    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    const words = std.mem.bytesAsSlice(u32, result.code);

    // Validate header (5 words)
    try std.testing.expect(words.len >= 5);

    // Word 0: Magic number
    try std.testing.expectEqual(SPIRV_MAGIC, words[0]);

    // Word 1: Version (1.5)
    try std.testing.expectEqual(SPIRV_VERSION, words[1]);

    // Word 2: Generator ID (should be 0 for unregistered)
    try std.testing.expectEqual(GENERATOR_ID, words[2]);

    // Word 3: Bound (upper bound of all IDs, must be > 0)
    try std.testing.expect(words[3] > 0);

    // Word 4: Schema (reserved, must be 0)
    try std.testing.expectEqual(@as(u32, 0), words[4]);

    // Verify first instruction after header is OpCapability (opcode 17)
    if (words.len > 5) {
        const first_op = words[5] & 0xFFFF;
        try std.testing.expectEqual(@as(u32, @intFromEnum(OpCode.OpCapability)), first_op);
    }

    // Validate that module has required sections in order:
    // 1. Capabilities (OpCapability)
    // 2. Memory model (OpMemoryModel)
    // 3. Entry points (OpEntryPoint)
    var found_capability = false;
    var found_memory_model = false;
    var found_entry_point = false;

    var i: usize = 5; // Start after header
    while (i < words.len) {
        const instruction = words[i];
        const word_count_inst = instruction >> 16;
        const op = instruction & 0xFFFF;

        if (op == @intFromEnum(OpCode.OpCapability)) found_capability = true;
        if (op == @intFromEnum(OpCode.OpMemoryModel)) found_memory_model = true;
        if (op == @intFromEnum(OpCode.OpEntryPoint)) found_entry_point = true;

        if (word_count_inst == 0) break;
        i += word_count_inst;
    }

    try std.testing.expect(found_capability);
    try std.testing.expect(found_memory_model);
    try std.testing.expect(found_entry_point);
}

test "SPIR-V external validation hint" {
    // This test documents how to validate SPIR-V externally with spirv-val
    // Users should run: spirv-val <output.spv> to validate generated shaders
    //
    // To generate a binary for validation:
    // 1. Use SpirvGenerator.generate() to get GeneratedSource
    // 2. Write result.code to a .spv file
    // 3. Run: spirv-val output.spv
    //
    // The spirv-val tool is part of the SPIRV-Tools package:
    // https://github.com/KhronosGroup/SPIRV-Tools
    //
    // Example validation command:
    //   spirv-val --target-env vulkan1.2 shader.spv
    //
    // This test just verifies the documentation exists and the generator works.
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = SpirvGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("external_val_test");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    // spirv_binary should be set for external validation
    try std.testing.expect(result.spirv_binary != null);
    try std.testing.expect(result.spirv_binary.?.len > 0);
}
