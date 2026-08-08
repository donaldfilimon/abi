// Metal DOT kernel shim for abi-gpu.
// Built on macOS when feature `metal-kernels` is enabled.
//
// Exports:
//   abi_metal_kernels_linked() -> Bool  (always true when this dylib loads)
//   abi_metal_available() -> Bool       (device present + pipeline ready)
//   abi_metal_dot(a, b, n) -> Float     (GPU DOT; NaN on failure)

import Foundation
import CoreML
import Metal

private final class MetalDotState: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let dotPipeline: MTLComputePipelineState
    let batchDotPipeline: MTLComputePipelineState
    let batchCosinePipeline: MTLComputePipelineState

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
        kernel void abi_batch_dot(
            device const float* query [[buffer(0)]],
            device const float* candidates [[buffer(1)]],
            device float* scores [[buffer(2)]],
            constant uint& dimensions [[buffer(3)]],
            constant uint& count [[buffer(4)]],
            uint candidate [[thread_position_in_grid]]
        ) {
            if (candidate >= count) { return; }
            float sum = 0.0f;
            uint offset = candidate * dimensions;
            for (uint index = 0; index < dimensions; index++) {
                sum += query[index] * candidates[offset + index];
            }
            scores[candidate] = sum;
        }
        kernel void abi_batch_cosine(
            device const float* query [[buffer(0)]],
            device const float* candidates [[buffer(1)]],
            device float* scores [[buffer(2)]],
            constant uint& dimensions [[buffer(3)]],
            constant uint& count [[buffer(4)]],
            uint candidate [[thread_position_in_grid]]
        ) {
            if (candidate >= count) { return; }
            float numerator = 0.0f;
            float querySquared = 0.0f;
            float candidateSquared = 0.0f;
            uint offset = candidate * dimensions;
            for (uint index = 0; index < dimensions; index++) {
                float left = query[index];
                float right = candidates[offset + index];
                numerator += left * right;
                querySquared += left * left;
                candidateSquared += right * right;
            }
            float denominator = sqrt(querySquared) * sqrt(candidateSquared);
            scores[candidate] = denominator == 0.0f ? 0.0f : numerator / denominator;
        }
        """
        guard let library = try? device.makeLibrary(source: source, options: nil) else {
            return nil
        }
        guard let dotFunction = library.makeFunction(name: "abi_dot"),
              let batchDotFunction = library.makeFunction(name: "abi_batch_dot"),
              let batchCosineFunction = library.makeFunction(name: "abi_batch_cosine"),
              let dotPipeline = try? device.makeComputePipelineState(function: dotFunction),
              let batchDotPipeline = try? device.makeComputePipelineState(
                  function: batchDotFunction
              ),
              let batchCosinePipeline = try? device.makeComputePipelineState(
                  function: batchCosineFunction
              )
        else {
            return nil
        }
        self.device = device
        self.queue = queue
        self.dotPipeline = dotPipeline
        self.batchDotPipeline = batchDotPipeline
        self.batchCosinePipeline = batchCosinePipeline
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
        enc.setComputePipelineState(dotPipeline)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(partialBuf, offset: 0, index: 2)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        // Power-of-two threadgroup size for the tree reduce (≤ 256).
        var tg = min(dotPipeline.maxTotalThreadsPerThreadgroup, 256)
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

    func batch(
        query: UnsafePointer<Float>, candidates: UnsafePointer<Float>,
        count: Int, dimensions: Int, cosine: Bool
    ) -> [Float]? {
        guard count > 0 else { return [] }
        guard dimensions > 0 else { return Array(repeating: 0, count: count) }
        let queryBytes = dimensions * MemoryLayout<Float>.stride
        let candidateBytes = count * queryBytes
        guard let queryBuffer = device.makeBuffer(
            bytes: query, length: queryBytes, options: .storageModeShared
        ), let candidateBuffer = device.makeBuffer(
            bytes: candidates, length: candidateBytes, options: .storageModeShared
        ), let scoreBuffer = device.makeBuffer(
            length: count * MemoryLayout<Float>.stride, options: .storageModeShared
        ) else { return nil }
        var dimensionValue = UInt32(dimensions)
        var countValue = UInt32(count)
        guard let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder()
        else { return nil }
        let pipeline = cosine ? batchCosinePipeline : batchDotPipeline
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidateBuffer, offset: 0, index: 1)
        encoder.setBuffer(scoreBuffer, offset: 0, index: 2)
        encoder.setBytes(
            &dimensionValue, length: MemoryLayout<UInt32>.stride, index: 3
        )
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 4)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { return nil }
        let pointer = scoreBuffer.contents().bindMemory(to: Float.self, capacity: count)
        let values = Array(UnsafeBufferPointer(start: pointer, count: count))
        return values.allSatisfy(\.isFinite) ? values : nil
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

@_cdecl("abi_metal_batch_dot")
public func abi_metal_batch_dot(
    _ query: UnsafePointer<Float>?, _ candidates: UnsafePointer<Float>?,
    _ count: Int, _ dimensions: Int, _ output: UnsafeMutablePointer<Float>?
) -> Bool {
    guard count >= 0, dimensions >= 0, let state, let output else { return false }
    if count == 0 { return true }
    guard let query, let candidates,
          let scores = state.batch(
              query: query, candidates: candidates, count: count,
              dimensions: dimensions, cosine: false
          )
    else { return false }
    for index in scores.indices { output[index] = scores[index] }
    return true
}

@_cdecl("abi_metal_cosine")
public func abi_metal_cosine(
    _ a: UnsafePointer<Float>?, _ b: UnsafePointer<Float>?, _ n: Int
) -> Float {
    guard let a, let b, n >= 0, let state else { return .nan }
    if n == 0 { return 0 }
    let numerator = state.dot(a: a, b: b, n: n)
    let leftNorm = sqrt(state.dot(a: a, b: a, n: n))
    let rightNorm = sqrt(state.dot(a: b, b: b, n: n))
    let denominator = leftNorm * rightNorm
    return denominator == 0 ? 0 : numerator / denominator
}

@_cdecl("abi_metal_norm")
public func abi_metal_norm(_ values: UnsafePointer<Float>?, _ n: Int) -> Float {
    guard let values, n >= 0, let state else { return .nan }
    if n == 0 { return 0 }
    return sqrt(state.dot(a: values, b: values, n: n))
}

@_cdecl("abi_metal_top_k")
public func abi_metal_top_k(
    _ query: UnsafePointer<Float>?, _ candidates: UnsafePointer<Float>?,
    _ count: Int, _ dimensions: Int, _ limit: Int,
    _ outputIndices: UnsafeMutablePointer<Int>?,
    _ outputScores: UnsafeMutablePointer<Float>?
) -> Int {
    guard count >= 0, dimensions >= 0, limit >= 0, let state,
          let outputIndices, let outputScores
    else { return -1 }
    if count == 0 || limit == 0 { return 0 }
    guard let query, let candidates,
          let scores = state.batch(
              query: query, candidates: candidates, count: count,
              dimensions: dimensions, cosine: true
          )
    else { return -1 }
    let ranked = scores.indices.sorted { left, right in
        scores[left] == scores[right] ? left < right : scores[left] > scores[right]
    }
    let returned = min(limit, count)
    for output in 0..<returned {
        outputIndices[output] = ranked[output]
        outputScores[output] = scores[ranked[output]]
    }
    return returned
}

@_cdecl("abi_coreml_cpu_and_neural_engine_requested")
public func abi_coreml_cpu_and_neural_engine_requested() -> Bool {
    if #available(macOS 12.0, *) {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        return configuration.computeUnits == .cpuAndNeuralEngine
    }
    return false
}
