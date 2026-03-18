import CABI

/// Main entry point for the ABI framework from Swift.
public final class ABI {
    private var framework: OpaquePointer?

    /// Initialize the ABI framework with default options.
    public init() throws {
        var ptr: OpaquePointer?
        let result = abi_init(&ptr)
        guard result == ABI_OK else {
            throw ABIError(code: result)
        }
        self.framework = ptr
    }

    deinit {
        if let fw = framework {
            abi_shutdown(fw)
        }
    }

    /// Get the framework version string.
    public static var version: String {
        String(cString: abi_version())
    }

    /// Check whether a compile-time feature is enabled.
    public func isFeatureEnabled(_ feature: String) -> Bool {
        abi_is_feature_enabled(framework, feature)
    }
}

/// Errors returned by ABI C functions, mapped to Swift.
public struct ABIError: Error, CustomStringConvertible {
    public let code: Int32

    public init(code: Int32) {
        self.code = code
    }

    public var description: String {
        String(cString: abi_error_string(code))
    }
}
