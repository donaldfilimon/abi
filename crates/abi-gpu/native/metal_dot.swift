// Metal DOT kernel shim for abi-gpu.
// Built on macOS when feature `metal-kernels` is enabled.
//
// Exports:
//   abi_metal_kernels_linked() -> Bool  (always true when this dylib loads)
//   abi_metal_available() -> Bool       (device present + pipeline ready)
//   abi_metal_dot(a, b, n) -> Float     (GPU DOT; NaN on failure)

import Foundation
import Metal

private final class MetalDotState: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLComputePipelineState

    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        guard let queue = device.makeCommandQueue() else { return nil }
        // Per-thread partial sums + tree reduce (exact under IEEE for pure sums
        // of products; avoids atomic_float non-determinism).
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void abi_dot(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* partial [[buffer(2)]],
            constant uint& n [[buffer(3)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]],
            uint tgId [[threadgroup_position_in_grid]]
        ) {
            threadgroup float scratch[256];
            float sum = 0.0f;
            for (uint i = tid; i < n; i += tgSize) {
                sum += a[i] * b[i];
            }
            scratch[tid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint offset = tgSize / 2; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    scratch[tid] += scratch[tid + offset];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                partial[tgId] = scratch[0];
            }
        }
        """
        guard let library = try? device.makeLibrary(source: source, options: nil) else {
            return nil
        }
        guard let function = library.makeFunction(name: "abi_dot") else { return nil }
        guard let pipeline = try? device.makeComputePipelineState(function: function) else {
            return nil
        }
        self.device = device
        self.queue = queue
        self.pipeline = pipeline
    }

    func dot(a: UnsafePointer<Float>, b: UnsafePointer<Float>, n: Int) -> Float {
        guard n > 0 else { return 0 }
        let byteLen = n * MemoryLayout<Float>.stride
        guard let aBuf = device.makeBuffer(
            bytes: a, length: byteLen, options: .storageModeShared
        ) else { return .nan }
        guard let bBuf = device.makeBuffer(
            bytes: b, length: byteLen, options: .storageModeShared
        ) else { return .nan }
        // One threadgroup → one partial sum.
        guard let partialBuf = device.makeBuffer(
            length: MemoryLayout<Float>.stride, options: .storageModeShared
        ) else { return .nan }
        partialBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = 0
        var count = UInt32(n)
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder()
        else { return .nan }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(partialBuf, offset: 0, index: 2)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        // Power-of-two threadgroup size for the tree reduce (≤ 256).
        var tg = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        // Round down to power of two.
        var pot = 1
        while pot * 2 <= tg { pot *= 2 }
        tg = pot
        let group = MTLSize(width: tg, height: 1, depth: 1)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: group
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if cmd.status != .completed {
            return .nan
        }
        return partialBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee
    }
}

private let state: MetalDotState? = MetalDotState()

@_cdecl("abi_metal_kernels_linked")
public func abi_metal_kernels_linked() -> Bool {
    true
}

@_cdecl("abi_metal_available")
public func abi_metal_available() -> Bool {
    state != nil
}

@_cdecl("abi_metal_dot")
public func abi_metal_dot(
    _ a: UnsafePointer<Float>?,
    _ b: UnsafePointer<Float>?,
    _ n: Int
) -> Float {
    guard let a, let b, n >= 0, let state else { return .nan }
    if n == 0 { return 0 }
    return state.dot(a: a, b: b, n: n)
}
