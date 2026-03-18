package com.abi

/**
 * Main entry point for the ABI framework from Kotlin/Android.
 *
 * Call [init] before using any other ABI functionality, and [shutdown]
 * when you are done.
 */
object ABI {
    init {
        System.loadLibrary("abi_jni")
    }

    /** Initialize the framework. Returns 0 on success. */
    external fun init(): Int

    /** Shut down the framework and release resources. */
    external fun shutdown()

    /** Get the framework version string. */
    external fun version(): String
}
