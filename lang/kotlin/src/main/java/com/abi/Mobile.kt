package com.abi

/**
 * Sensor data returned by [MobileContext.readSensor].
 */
data class SensorData(
    val timestampMs: Long,
    val values: FloatArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SensorData) return false
        return timestampMs == other.timestampMs && values.contentEquals(other.values)
    }

    override fun hashCode(): Int {
        var result = timestampMs.hashCode()
        result = 31 * result + values.contentHashCode()
        return result
    }
}

/**
 * Device information returned by [MobileContext.getDeviceInfo].
 */
data class DeviceInfo(
    val screenWidth: Int,
    val screenHeight: Int,
    val batteryLevel: Float,
    val isCharging: Boolean,
    val platform: String,
    val osVersion: String,
    val deviceModel: String,
)

/** Sensor type constants. */
object SensorType {
    const val ACCELEROMETER = 0
    const val GYROSCOPE     = 1
    const val MAGNETOMETER  = 2
    const val GPS           = 3
    const val BAROMETER     = 4
    const val PROXIMITY     = 5
    const val LIGHT         = 6
}

/** Permission type constants. */
object PermissionType {
    const val CAMERA        = 0
    const val MICROPHONE    = 1
    const val LOCATION      = 2
    const val NOTIFICATIONS = 3
    const val STORAGE       = 4
    const val CONTACTS      = 5
    const val BLUETOOTH     = 6
}

/** Permission status constants. */
object PermissionStatus {
    const val GRANTED       = 0
    const val DENIED        = 1
    const val NOT_REQUESTED = 2
}

/** Notification priority constants. */
object NotificationPriority {
    const val LOW      = 0
    const val NORMAL   = 1
    const val HIGH     = 2
    const val CRITICAL = 3
}

/**
 * Kotlin wrapper around the ABI mobile C API via JNI.
 *
 * Create an instance with [MobileContext] and call [close] when finished.
 * Implements [AutoCloseable] so it can be used with `use {}`.
 */
class MobileContext : AutoCloseable {
    /** Native pointer held by JNI. */
    private var nativePtr: Long = 0L

    init {
        nativePtr = nativeInit()
        if (nativePtr == 0L) {
            throw RuntimeException("Failed to initialize MobileContext")
        }
    }

    /** Read a sensor value. */
    fun readSensor(sensorType: Int): SensorData = nativeReadSensor(nativePtr, sensorType)

    /** Send a notification. */
    fun sendNotification(
        title: String,
        body: String,
        priority: Int = NotificationPriority.NORMAL,
    ): Int = nativeSendNotification(nativePtr, title, body, priority)

    /** Get device information. */
    fun getDeviceInfo(): DeviceInfo = nativeGetDeviceInfo(nativePtr)

    /** Check the status of a permission. */
    fun checkPermission(permission: Int): Int = nativeCheckPermission(nativePtr, permission)

    /** Request a permission. */
    fun requestPermission(permission: Int): Int = nativeRequestPermission(nativePtr, permission)

    /** Number of tracked notifications. */
    val notificationCount: Int get() = nativeGetNotificationCount(nativePtr)

    /** Clear all tracked notifications. */
    fun clearNotifications() = nativeClearNotifications(nativePtr)

    override fun close() {
        if (nativePtr != 0L) {
            nativeDestroy(nativePtr)
            nativePtr = 0L
        }
    }

    // JNI native methods
    private external fun nativeInit(): Long
    private external fun nativeDestroy(ptr: Long)
    private external fun nativeReadSensor(ptr: Long, sensorType: Int): SensorData
    private external fun nativeSendNotification(ptr: Long, title: String, body: String, priority: Int): Int
    private external fun nativeGetDeviceInfo(ptr: Long): DeviceInfo
    private external fun nativeCheckPermission(ptr: Long, permission: Int): Int
    private external fun nativeRequestPermission(ptr: Long, permission: Int): Int
    private external fun nativeGetNotificationCount(ptr: Long): Int
    private external fun nativeClearNotifications(ptr: Long)

    companion object {
        init {
            System.loadLibrary("abi_jni")
        }
    }
}
