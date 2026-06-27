// Apple FoundationModels on-device bridge for the ABI Zig connector.
//
// FoundationModels.framework is a *Swift-only* framework: it exports no
// Objective-C classes, so its `SystemLanguageModel` / `LanguageModelSession`
// types are unreachable from `objc_getClass` / `objc_msgSend`. This file is the
// thin Swift shim that exposes a synchronous C ABI the Zig side can call
// (`src/connectors/fm.zig`), using the official `@c` attribute (SE-0495, Swift
// 6.3+) rather than the underscored `@_cdecl`. It is compiled and linked ONLY on
// macOS with `-Dfeat-foundationmodels`; the default ABI build never sees it.
//
// Return-code contract (shared with fm.zig):
//   abi_fm_available() -> Int32
//     1  = model present and ready (Apple Intelligence enabled + model loaded)
//     0  = framework reachable but model not available (not enabled / not
//          downloaded / unsupported device)
//    -1  = OS too old (FoundationModels API not available at runtime)
//
//   abi_fm_complete(prompt, out, out_len) -> Int32
//     >=0 = success; the value is the number of UTF-8 bytes written into `out`
//           (NUL-terminated, truncated to out_len-1 if needed)
//      -1 = null argument (prompt or out was nil, or out_len < 1)
//      -2 = OS too old (FoundationModels API not available at runtime)
//      -3 = model unavailable (not enabled / not downloaded / unsupported)
//      -4 = generation error (session threw while producing a response)

import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Copy a Swift string into the caller's C buffer, always NUL-terminating and
/// truncating on a UTF-8 boundary. Returns the number of bytes written (excluding
/// the terminator).
@inline(__always)
private func copyOut(_ s: String, _ out: UnsafeMutablePointer<CChar>, _ outLen: Int) -> Int32 {
    if outLen < 1 { return -1 }
    let bytes = Array(s.utf8)
    let maxCopy = min(bytes.count, outLen - 1)
    var written = 0
    bytes.withUnsafeBufferPointer { buf in
        var i = 0
        while i < maxCopy {
            out[i] = CChar(bitPattern: buf[i])
            i += 1
        }
        written = maxCopy
    }
    out[written] = 0
    return Int32(written)
}

@c(abi_fm_available)
public func abi_fm_available() -> Int32 {
#if canImport(FoundationModels)
    if #available(macOS 26, *) {
        switch SystemLanguageModel.default.availability {
        case .available:
            return 1
        case .unavailable:
            return 0
        @unknown default:
            return 0
        }
    }
    return -1
#else
    return -1
#endif
}

@c(abi_fm_complete)
public func abi_fm_complete(
    prompt: UnsafePointer<CChar>?,
    out: UnsafeMutablePointer<CChar>?,
    out_len: Int
) -> Int32 {
    guard let promptPtr = prompt, let outPtr = out, out_len > 0 else {
        return -1
    }
#if canImport(FoundationModels)
    if #available(macOS 26, *) {
        switch SystemLanguageModel.default.availability {
        case .available:
            break
        case .unavailable:
            return -3
        @unknown default:
            return -3
        }

        let promptString = String(cString: promptPtr)

        // Drive the async `respond(to:)` to completion synchronously: the Zig
        // caller expects a blocking C function. A synchronous C ABI boundary has
        // no other option, so the Task + DispatchSemaphore.wait() bridge is the
        // justified pattern here (the usual "prefer Task, avoid semaphores"
        // guidance is about ordinary async code, not a sync entry point). We hop
        // onto a detached Task and park the calling thread until the result lands.
        let semaphore = DispatchSemaphore(value: 0)
        let resultBox = ResultBox()

        Task {
            do {
                let session = LanguageModelSession()
                let response = try await session.respond(to: promptString)
                resultBox.text = response.content
            } catch {
                resultBox.failed = true
            }
            semaphore.signal()
        }
        semaphore.wait()

        if resultBox.failed {
            return -4
        }
        guard let text = resultBox.text else {
            return -4
        }
        return copyOut(text, outPtr, out_len)
    }
    return -2
#else
    return -2
#endif
}

/// Reference box so the detached Task can hand the result back across the
/// semaphore boundary without capturing an inout.
private final class ResultBox: @unchecked Sendable {
    var text: String?
    var failed: Bool = false
}
