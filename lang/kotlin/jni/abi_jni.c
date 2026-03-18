/**
 * JNI glue for the ABI mobile C API.
 *
 * This file bridges Kotlin's MobileContext class to the abi_mobile_* C
 * functions. The native pointer is stored as a Java long on the Kotlin side.
 */

#include <jni.h>
#include <string.h>
#include "../../../bindings/c/include/abi.h"

/* ========================================================================= */
/* ABI object                                                                */
/* ========================================================================= */

JNIEXPORT jint JNICALL
Java_com_abi_ABI_init(JNIEnv *env, jobject thiz)
{
    (void)env; (void)thiz;
    /* Framework-level init could be added here in future. */
    return ABI_OK;
}

JNIEXPORT void JNICALL
Java_com_abi_ABI_shutdown(JNIEnv *env, jobject thiz)
{
    (void)env; (void)thiz;
}

JNIEXPORT jstring JNICALL
Java_com_abi_ABI_version(JNIEnv *env, jobject thiz)
{
    (void)thiz;
    return (*env)->NewStringUTF(env, abi_version());
}

/* ========================================================================= */
/* MobileContext                                                             */
/* ========================================================================= */

JNIEXPORT jlong JNICALL
Java_com_abi_MobileContext_nativeInit(JNIEnv *env, jobject thiz)
{
    (void)env; (void)thiz;
    abi_mobile_context_t *ctx = NULL;
    int rc = abi_mobile_init(&ctx);
    if (rc != ABI_OK) return 0;
    return (jlong)(intptr_t)ctx;
}

JNIEXPORT void JNICALL
Java_com_abi_MobileContext_nativeDestroy(JNIEnv *env, jobject thiz,
                                         jlong ptr)
{
    (void)env; (void)thiz;
    if (ptr != 0) {
        abi_mobile_destroy((abi_mobile_context_t *)(intptr_t)ptr);
    }
}

JNIEXPORT jobject JNICALL
Java_com_abi_MobileContext_nativeReadSensor(JNIEnv *env, jobject thiz,
                                            jlong ptr, jint sensor_type)
{
    (void)thiz;
    abi_mobile_context_t *ctx = (abi_mobile_context_t *)(intptr_t)ptr;
    abi_sensor_data_t data;
    memset(&data, 0, sizeof(data));

    int rc = abi_mobile_read_sensor(ctx, sensor_type, &data);
    if (rc != ABI_OK) return NULL;

    /* Build a float[] for the values. */
    jfloatArray values = (*env)->NewFloatArray(env, 3);
    (*env)->SetFloatArrayRegion(env, values, 0, 3, data.values);

    /* Construct SensorData(timestampMs, values). */
    jclass cls = (*env)->FindClass(env, "com/abi/SensorData");
    jmethodID ctor = (*env)->GetMethodID(env, cls, "<init>", "(J[F)V");
    return (*env)->NewObject(env, cls, ctor, (jlong)data.timestamp_ms, values);
}

JNIEXPORT jint JNICALL
Java_com_abi_MobileContext_nativeSendNotification(JNIEnv *env, jobject thiz,
                                                   jlong ptr,
                                                   jstring title,
                                                   jstring body,
                                                   jint priority)
{
    (void)thiz;
    abi_mobile_context_t *ctx = (abi_mobile_context_t *)(intptr_t)ptr;

    const char *c_title = (*env)->GetStringUTFChars(env, title, NULL);
    const char *c_body  = (*env)->GetStringUTFChars(env, body, NULL);

    int rc = abi_mobile_send_notification(ctx, c_title, c_body, priority);

    (*env)->ReleaseStringUTFChars(env, title, c_title);
    (*env)->ReleaseStringUTFChars(env, body, c_body);

    return rc;
}

JNIEXPORT jobject JNICALL
Java_com_abi_MobileContext_nativeGetDeviceInfo(JNIEnv *env, jobject thiz,
                                               jlong ptr)
{
    (void)thiz;
    abi_mobile_context_t *ctx = (abi_mobile_context_t *)(intptr_t)ptr;
    abi_device_info_t info;
    memset(&info, 0, sizeof(info));

    int rc = abi_mobile_get_device_info(ctx, &info);
    if (rc != ABI_OK) return NULL;

    jstring platform    = (*env)->NewStringUTF(env, info.platform);
    jstring os_version  = (*env)->NewStringUTF(env, info.os_version);
    jstring device_model = (*env)->NewStringUTF(env, info.device_model);

    /* DeviceInfo(screenWidth, screenHeight, batteryLevel, isCharging,
     *           platform, osVersion, deviceModel) */
    jclass cls = (*env)->FindClass(env, "com/abi/DeviceInfo");
    jmethodID ctor = (*env)->GetMethodID(
        env, cls, "<init>",
        "(IIFZLjava/lang/String;Ljava/lang/String;Ljava/lang/String;)V");
    return (*env)->NewObject(env, cls, ctor,
                             (jint)info.screen_width,
                             (jint)info.screen_height,
                             (jfloat)info.battery_level,
                             (jboolean)info.is_charging,
                             platform, os_version, device_model);
}

JNIEXPORT jint JNICALL
Java_com_abi_MobileContext_nativeCheckPermission(JNIEnv *env, jobject thiz,
                                                  jlong ptr, jint permission)
{
    (void)env; (void)thiz;
    abi_mobile_context_t *ctx = (abi_mobile_context_t *)(intptr_t)ptr;
    return abi_mobile_check_permission(ctx, permission);
}

JNIEXPORT jint JNICALL
Java_com_abi_MobileContext_nativeRequestPermission(JNIEnv *env, jobject thiz,
                                                    jlong ptr, jint permission)
{
    (void)env; (void)thiz;
    abi_mobile_context_t *ctx = (abi_mobile_context_t *)(intptr_t)ptr;
    return abi_mobile_request_permission(ctx, permission);
}

JNIEXPORT jint JNICALL
Java_com_abi_MobileContext_nativeGetNotificationCount(JNIEnv *env,
                                                      jobject thiz,
                                                      jlong ptr)
{
    (void)env; (void)thiz;
    abi_mobile_context_t *ctx = (abi_mobile_context_t *)(intptr_t)ptr;
    return abi_mobile_get_notification_count(ctx);
}

JNIEXPORT void JNICALL
Java_com_abi_MobileContext_nativeClearNotifications(JNIEnv *env, jobject thiz,
                                                    jlong ptr)
{
    (void)env; (void)thiz;
    abi_mobile_context_t *ctx = (abi_mobile_context_t *)(intptr_t)ptr;
    abi_mobile_clear_notifications(ctx);
}
