# Language Bindings

High-level language bindings for the ABI framework.

## Available Bindings

### Swift (`lang/swift/`)

Swift Package Manager library targeting iOS 15+ and macOS 13+.

**Supported platforms:** iOS, macOS (any platform with Swift 5.9+).

**Setup:**

Add the package as a local dependency in your `Package.swift`:

```swift
.package(path: "/path/to/abi/lang/swift")
```

Or reference it as a dependency in Xcode via *File > Add Package Dependencies*.

**Usage:**

```swift
import ABI

// Framework
let framework = try ABI()
print("Version: \(ABI.version)")

// Mobile
let mobile = try MobileContext()
let accel = try mobile.readSensor(.accelerometer)
print("Accel: \(accel.values)")

let info = try mobile.getDeviceInfo()
print("Device: \(info.deviceModel)")

mobile.requestPermission(.camera)
try mobile.sendNotification(title: "Hello", body: "World", priority: .high)
```

**Build requirements:** The `libabi` static/shared library must be built
first (`zig build`) and available on the linker search path.

---

### Kotlin/JNI (`lang/kotlin/`)

Android library module with JNI glue for the ABI C API.

**Supported platforms:** Android (minSdk 26, NDK required).

**Gradle setup:**

Include the module in your Android project's `settings.gradle.kts`:

```kotlin
include(":abi")
project(":abi").projectDir = File("/path/to/abi/lang/kotlin")
```

Then add the dependency:

```kotlin
dependencies {
    implementation(project(":abi"))
}
```

**Usage:**

```kotlin
import com.abi.*

// Framework
ABI.init()
println("Version: ${ABI.version()}")

// Mobile
MobileContext().use { mobile ->
    val accel = mobile.readSensor(SensorType.ACCELEROMETER)
    println("Accel: ${accel.values.toList()}")

    val info = mobile.getDeviceInfo()
    println("Device: ${info.deviceModel}")

    mobile.requestPermission(PermissionType.CAMERA)
    mobile.sendNotification("Hello", "World", NotificationPriority.HIGH)
}

ABI.shutdown()
```

**Build requirements:** Android NDK, CMake 3.22+, and the pre-built `libabi`
shared library placed in `zig-out/lib/` (or adjust the path in
`build.gradle.kts`).

---

## Existing Low-Level Bindings

The `bindings/` directory already provides two integration surfaces:

- **C** (`bindings/c/`) -- C header and source files exposing the core API,
  including the plugin registry (`abi_plugin_register`, etc.). Use these to
  embed ABI in any language with a C FFI.
- **WASM** (`bindings/wasm/`) -- WebAssembly target for browser and edge
  runtimes.

High-level bindings in this directory will typically wrap the C bindings with
idiomatic APIs for their respective language ecosystems.

## Adding a New Binding

1. Create a subdirectory named after the target language (e.g., `lang/python/`).
2. Provide a build script or manifest appropriate for that ecosystem.
3. Wrap the C API from `bindings/c/include/abi.h`.
4. Include a `README.md` with setup and usage instructions.
