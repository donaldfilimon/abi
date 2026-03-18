import CABI

/// Idiomatic Swift wrapper around the ABI mobile C API.
public final class MobileContext {
    private var handle: OpaquePointer?

    /// Initialize a mobile context.
    public init() throws {
        var ptr: OpaquePointer?
        let result = abi_mobile_init(&ptr)
        guard result == ABI_OK else {
            throw ABIError(code: result)
        }
        self.handle = ptr
    }

    deinit {
        if let h = handle {
            abi_mobile_destroy(h)
        }
    }

    // MARK: - Sensors

    /// Sensor type constants matching the C API.
    public enum SensorType: Int32 {
        case accelerometer  = 0
        case gyroscope      = 1
        case magnetometer   = 2
        case gps            = 3
        case barometer      = 4
        case proximity      = 5
        case light          = 6
    }

    /// Data returned from a sensor reading.
    public struct SensorData {
        public let timestampMs: UInt64
        public let values: (Float, Float, Float)
    }

    /// Read a sensor value.
    public func readSensor(_ type: SensorType) throws -> SensorData {
        var raw = abi_sensor_data_t()
        let result = abi_mobile_read_sensor(handle, type.rawValue, &raw)
        guard result == ABI_OK else {
            throw ABIError(code: result)
        }
        return SensorData(
            timestampMs: raw.timestamp_ms,
            values: (raw.values.0, raw.values.1, raw.values.2)
        )
    }

    // MARK: - Notifications

    /// Notification priority levels.
    public enum NotificationPriority: Int32 {
        case low      = 0
        case normal   = 1
        case high     = 2
        case critical = 3
    }

    /// Send a notification.
    public func sendNotification(title: String, body: String,
                                 priority: NotificationPriority = .normal) throws {
        let result = abi_mobile_send_notification(handle, title, body, priority.rawValue)
        guard result == ABI_OK else {
            throw ABIError(code: result)
        }
    }

    /// The number of tracked notifications.
    public var notificationCount: Int {
        Int(abi_mobile_get_notification_count(handle))
    }

    /// Clear all tracked notifications.
    public func clearNotifications() {
        abi_mobile_clear_notifications(handle)
    }

    // MARK: - Device Info

    /// Device information snapshot.
    public struct DeviceInfo {
        public let screenWidth: UInt32
        public let screenHeight: UInt32
        public let batteryLevel: Float
        public let isCharging: Bool
        public let platform: String
        public let osVersion: String
        public let deviceModel: String
    }

    /// Get current device information.
    public func getDeviceInfo() throws -> DeviceInfo {
        var raw = abi_device_info_t()
        let result = abi_mobile_get_device_info(handle, &raw)
        guard result == ABI_OK else {
            throw ABIError(code: result)
        }
        return DeviceInfo(
            screenWidth: raw.screen_width,
            screenHeight: raw.screen_height,
            batteryLevel: raw.battery_level,
            isCharging: raw.is_charging,
            platform: String(cString: raw.platform),
            osVersion: String(cString: raw.os_version),
            deviceModel: String(cString: raw.device_model)
        )
    }

    // MARK: - Permissions

    /// Permission types.
    public enum Permission: Int32 {
        case camera        = 0
        case microphone    = 1
        case location      = 2
        case notifications = 3
        case storage       = 4
        case contacts      = 5
        case bluetooth     = 6
    }

    /// Permission status values.
    public enum PermissionStatus: Int32 {
        case granted      = 0
        case denied       = 1
        case notRequested = 2
    }

    /// Check the current status of a permission.
    public func checkPermission(_ permission: Permission) -> PermissionStatus {
        let raw = abi_mobile_check_permission(handle, permission.rawValue)
        return PermissionStatus(rawValue: raw) ?? .notRequested
    }

    /// Request a permission.
    @discardableResult
    public func requestPermission(_ permission: Permission) -> PermissionStatus {
        let raw = abi_mobile_request_permission(handle, permission.rawValue)
        return PermissionStatus(rawValue: raw) ?? .notRequested
    }
}
