package abi

/*
#include <stdlib.h>
#include <stdbool.h>

typedef int abi_error_t;
typedef void* abi_gpu_t;

typedef struct {
    int backend;
    int device_index;
} abi_gpu_config_t;

typedef struct {
    char name[256];
    int backend;
    size_t total_memory;
    size_t free_memory;
    int compute_units;
} abi_gpu_device_info_t;

extern abi_error_t abi_gpu_init(const abi_gpu_config_t* config, abi_gpu_t* out_gpu);
extern void abi_gpu_shutdown(abi_gpu_t gpu);
extern bool abi_gpu_is_available();
extern abi_error_t abi_gpu_list_devices(abi_gpu_device_info_t* out_devices,
                                        size_t max_devices, size_t* out_count);
extern abi_error_t abi_gpu_matrix_multiply(abi_gpu_t gpu,
                                           const float* a, const float* b, float* result,
                                           size_t m, size_t n, size_t k);
extern abi_error_t abi_gpu_vector_add(abi_gpu_t gpu,
                                      const float* a, const float* b, float* result, size_t len);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// GPUBackend represents a GPU compute backend
type GPUBackend int

const (
	BackendAuto GPUBackend = iota
	BackendVulkan
	BackendCUDA
	BackendMetal
	BackendWebGPU
	BackendOpenGL
	BackendCPU
)

func (b GPUBackend) String() string {
	switch b {
	case BackendVulkan:
		return "Vulkan"
	case BackendCUDA:
		return "CUDA"
	case BackendMetal:
		return "Metal"
	case BackendWebGPU:
		return "WebGPU"
	case BackendOpenGL:
		return "OpenGL"
	case BackendCPU:
		return "CPU"
	default:
		return "Auto"
	}
}

// GPUDeviceInfo contains information about a GPU device
type GPUDeviceInfo struct {
	Name         string
	Backend      GPUBackend
	TotalMemory  uint64
	FreeMemory   uint64
	ComputeUnits int
}

// GPUConfig contains configuration for creating a GPU context
type GPUConfig struct {
	Backend     GPUBackend
	DeviceIndex int
}

// DefaultGPUConfig returns a configuration with auto-detected backend
func DefaultGPUConfig() GPUConfig {
	return GPUConfig{
		Backend:     BackendAuto,
		DeviceIndex: 0,
	}
}

// WithBackend sets the GPU backend
func (c GPUConfig) WithBackend(backend GPUBackend) GPUConfig {
	c.Backend = backend
	return c
}

// WithDevice sets the device index
func (c GPUConfig) WithDevice(index int) GPUConfig {
	c.DeviceIndex = index
	return c
}

// GPUAvailable returns true if any GPU is available
func GPUAvailable() bool {
	return bool(C.abi_gpu_is_available())
}

// ListGPUDevices returns information about all available GPU devices
func ListGPUDevices() ([]GPUDeviceInfo, error) {
	devices := make([]C.abi_gpu_device_info_t, 16)
	var count C.size_t

	code := C.abi_gpu_list_devices(
		(*C.abi_gpu_device_info_t)(unsafe.Pointer(&devices[0])),
		C.size_t(len(devices)),
		&count,
	)
	if err := convertError(code); err != nil {
		return nil, err
	}

	result := make([]GPUDeviceInfo, int(count))
	for i := 0; i < int(count); i++ {
		d := devices[i]
		result[i] = GPUDeviceInfo{
			Name:         C.GoString(&d.name[0]),
			Backend:      GPUBackend(d.backend),
			TotalMemory:  uint64(d.total_memory),
			FreeMemory:   uint64(d.free_memory),
			ComputeUnits: int(d.compute_units),
		}
	}

	return result, nil
}

// GPUContext provides GPU-accelerated computations
type GPUContext struct {
	handle C.abi_gpu_t
}

// NewGPUContext creates a new GPU context with the specified backend
func NewGPUContext(backend GPUBackend) (*GPUContext, error) {
	return NewGPUContextWithConfig(GPUConfig{Backend: backend, DeviceIndex: 0})
}

// NewGPUContextWithConfig creates a new GPU context with custom configuration
func NewGPUContextWithConfig(config GPUConfig) (*GPUContext, error) {
	cConfig := C.abi_gpu_config_t{
		backend:      C.int(config.Backend),
		device_index: C.int(config.DeviceIndex),
	}

	var handle C.abi_gpu_t
	code := C.abi_gpu_init(&cConfig, &handle)
	if err := convertError(code); err != nil {
		return nil, fmt.Errorf("failed to initialize GPU: %w", err)
	}

	return &GPUContext{handle: handle}, nil
}

// Close releases GPU resources
func (g *GPUContext) Close() {
	if g.handle != nil {
		C.abi_gpu_shutdown(g.handle)
		g.handle = nil
	}
}

// MatrixMultiply performs GPU-accelerated matrix multiplication: C = A * B
// A is m x k, B is k x n, result is m x n (row-major order)
func (g *GPUContext) MatrixMultiply(a, b []float32, m, n, k int) ([]float32, error) {
	if len(a) != m*k {
		return nil, fmt.Errorf("%w: matrix A has wrong size", ErrInvalidArgument)
	}
	if len(b) != k*n {
		return nil, fmt.Errorf("%w: matrix B has wrong size", ErrInvalidArgument)
	}

	result := make([]float32, m*n)

	code := C.abi_gpu_matrix_multiply(
		g.handle,
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&result[0])),
		C.size_t(m),
		C.size_t(n),
		C.size_t(k),
	)
	if err := convertError(code); err != nil {
		return nil, err
	}

	return result, nil
}

// VectorAdd performs GPU-accelerated vector addition: result[i] = a[i] + b[i]
func (g *GPUContext) VectorAdd(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("%w: vectors have different lengths", ErrInvalidArgument)
	}

	result := make([]float32, len(a))

	code := C.abi_gpu_vector_add(
		g.handle,
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&result[0])),
		C.size_t(len(a)),
	)
	if err := convertError(code); err != nil {
		return nil, err
	}

	return result, nil
}
