// Package abi provides Go bindings for the ABI Framework.
//
// The ABI Framework is a high-performance vector database and AI inference library
// written in Zig. These bindings use cgo to interface with the C shared library.
//
// # Requirements
//
// Build the shared library first:
//
//	zig build lib
//
// Then set the library path:
//
//	export LD_LIBRARY_PATH=$PWD/zig-out/lib:$LD_LIBRARY_PATH  # Linux
//	export DYLD_LIBRARY_PATH=$PWD/zig-out/lib:$DYLD_LIBRARY_PATH  # macOS
//
// # Example
//
//	package main
//
//	import (
//		"fmt"
//		"github.com/donaldfilimon/abi/bindings/go"
//	)
//
//	func main() {
//		fw, err := abi.Init()
//		if err != nil {
//			panic(err)
//		}
//		defer fw.Shutdown()
//
//		fmt.Println("ABI Version:", abi.Version())
//
//		db, err := abi.CreateDatabase(nil)
//		if err != nil {
//			panic(err)
//		}
//		defer db.Close()
//	}
package abi

/*
#cgo LDFLAGS: -labi
#cgo CFLAGS: -I${SRCDIR}/../c/include

#include <abi.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Error codes matching C API (negative integers).
const (
	OK                    = 0
	ErrInitFailed         = -1
	ErrAlreadyInitialized = -2
	ErrNotInitialized     = -3
	ErrOutOfMemory        = -4
	ErrInvalidArgument    = -5
	ErrFeatureDisabled    = -6
	ErrTimeout            = -7
	ErrIO                 = -8
	ErrGPUUnavailable     = -9
	ErrDatabaseError      = -10
	ErrNetworkError       = -11
	ErrAIError            = -12
	ErrUnknown            = -99
)

// Go error values.
var (
	ErrInit        = errors.New("abi: initialization failed")
	ErrAlreadyInit = errors.New("abi: already initialized")
	ErrNotInit     = errors.New("abi: not initialized")
	ErrOOM         = errors.New("abi: out of memory")
	ErrInvalidArg  = errors.New("abi: invalid argument")
	ErrDisabled    = errors.New("abi: feature disabled")
	ErrTimeoutErr  = errors.New("abi: timeout")
	ErrIOErr       = errors.New("abi: I/O error")
	ErrGPU         = errors.New("abi: GPU unavailable")
	ErrDatabase    = errors.New("abi: database error")
	ErrNetwork     = errors.New("abi: network error")
	ErrAI          = errors.New("abi: AI error")
	ErrUnknownErr  = errors.New("abi: unknown error")
)

// codeToError converts a C error code to a Go error.
func codeToError(code C.int) error {
	switch int(code) {
	case OK:
		return nil
	case ErrInitFailed:
		return ErrInit
	case ErrAlreadyInitialized:
		return ErrAlreadyInit
	case ErrNotInitialized:
		return ErrNotInit
	case ErrOutOfMemory:
		return ErrOOM
	case ErrInvalidArgument:
		return ErrInvalidArg
	case ErrFeatureDisabled:
		return ErrDisabled
	case ErrTimeout:
		return ErrTimeoutErr
	case ErrIO:
		return ErrIOErr
	case ErrGPUUnavailable:
		return ErrGPU
	case ErrDatabaseError:
		return ErrDatabase
	case ErrNetworkError:
		return ErrNetwork
	case ErrAIError:
		return ErrAI
	default:
		return ErrUnknownErr
	}
}

// ErrorString returns a human-readable error message for an error code.
func ErrorString(code int) string {
	return C.GoString(C.abi_error_string(C.int(code)))
}

// ============================================================================
// Framework
// ============================================================================

// Framework represents the ABI framework instance.
type Framework struct {
	handle *C.abi_framework_t
}

// Init initializes the ABI framework with default options.
func Init() (*Framework, error) {
	var handle *C.abi_framework_t
	code := C.abi_init(&handle)
	if err := codeToError(code); err != nil {
		return nil, err
	}
	return &Framework{handle: handle}, nil
}

// Options configures framework initialization.
type Options struct {
	EnableAI        bool
	EnableGPU       bool
	EnableDatabase  bool
	EnableNetwork   bool
	EnableWeb       bool
	EnableProfiling bool
}

// DefaultOptions returns options with all features enabled.
func DefaultOptions() Options {
	return Options{
		EnableAI:        true,
		EnableGPU:       true,
		EnableDatabase:  true,
		EnableNetwork:   true,
		EnableWeb:       true,
		EnableProfiling: true,
	}
}

// InitWithOptions initializes the ABI framework with custom options.
func InitWithOptions(opts Options) (*Framework, error) {
	cOpts := C.abi_options_t{
		enable_ai:        C.bool(opts.EnableAI),
		enable_gpu:       C.bool(opts.EnableGPU),
		enable_database:  C.bool(opts.EnableDatabase),
		enable_network:   C.bool(opts.EnableNetwork),
		enable_web:       C.bool(opts.EnableWeb),
		enable_profiling: C.bool(opts.EnableProfiling),
	}
	var handle *C.abi_framework_t
	code := C.abi_init_with_options(&cOpts, &handle)
	if err := codeToError(code); err != nil {
		return nil, err
	}
	return &Framework{handle: handle}, nil
}

// Shutdown releases all resources associated with the framework.
func (f *Framework) Shutdown() {
	if f.handle != nil {
		C.abi_shutdown(f.handle)
		f.handle = nil
	}
}

// IsFeatureEnabled checks if a feature is enabled.
// Valid features: "ai", "gpu", "database", "network", "web", "profiling".
func (f *Framework) IsFeatureEnabled(feature string) bool {
	cFeature := C.CString(feature)
	defer C.free(unsafe.Pointer(cFeature))
	return bool(C.abi_is_feature_enabled(f.handle, cFeature))
}

// Version returns the ABI framework version string.
func Version() string {
	return C.GoString(C.abi_version())
}

// VersionInfo holds detailed version information.
type VersionInfo struct {
	Major int
	Minor int
	Patch int
	Full  string
}

// GetVersionInfo returns detailed version information.
func GetVersionInfo() VersionInfo {
	var info C.abi_version_info_t
	C.abi_version_info(&info)
	return VersionInfo{
		Major: int(info.major),
		Minor: int(info.minor),
		Patch: int(info.patch),
		Full:  C.GoString(info.full),
	}
}

// ============================================================================
// SIMD
// ============================================================================

// SimdCaps describes CPU SIMD capabilities.
type SimdCaps struct {
	SSE    bool
	SSE2   bool
	SSE3   bool
	SSSE3  bool
	SSE4_1 bool
	SSE4_2 bool
	AVX    bool
	AVX2   bool
	AVX512F bool
	NEON   bool
}

// SimdAvailable returns true if any SIMD instruction set is available.
func SimdAvailable() bool {
	return bool(C.abi_simd_available())
}

// SimdGetCaps queries CPU SIMD capabilities.
func SimdGetCaps() SimdCaps {
	var caps C.abi_simd_caps_t
	C.abi_simd_get_caps(&caps)
	return SimdCaps{
		SSE:     bool(caps.sse),
		SSE2:    bool(caps.sse2),
		SSE3:    bool(caps.sse3),
		SSSE3:   bool(caps.ssse3),
		SSE4_1:  bool(caps.sse4_1),
		SSE4_2:  bool(caps.sse4_2),
		AVX:     bool(caps.avx),
		AVX2:    bool(caps.avx2),
		AVX512F: bool(caps.avx512f),
		NEON:    bool(caps.neon),
	}
}

// SimdVectorAdd computes element-wise addition: result[i] = a[i] + b[i].
func SimdVectorAdd(a, b []float32) []float32 {
	n := len(a)
	result := make([]float32, n)
	C.abi_simd_vector_add(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&result[0])),
		C.size_t(n),
	)
	return result
}

// SimdVectorDot computes the dot product of two vectors.
func SimdVectorDot(a, b []float32) float32 {
	return float32(C.abi_simd_vector_dot(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.size_t(len(a)),
	))
}

// SimdVectorL2Norm computes the L2 norm of a vector.
func SimdVectorL2Norm(v []float32) float32 {
	return float32(C.abi_simd_vector_l2_norm(
		(*C.float)(unsafe.Pointer(&v[0])),
		C.size_t(len(v)),
	))
}

// SimdCosineSimilarity computes the cosine similarity between two vectors.
func SimdCosineSimilarity(a, b []float32) float32 {
	return float32(C.abi_simd_cosine_similarity(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.size_t(len(a)),
	))
}

// ============================================================================
// Database
// ============================================================================

// DatabaseConfig configures a vector database.
type DatabaseConfig struct {
	Name            string
	Dimension       uint
	InitialCapacity uint
}

// VectorDatabase represents a vector database instance.
type VectorDatabase struct {
	handle *C.abi_database_t
}

// CreateDatabase creates a new vector database.
// Pass nil for default config (name="default", dimension=384, capacity=1000).
func CreateDatabase(config *DatabaseConfig) (*VectorDatabase, error) {
	var handle *C.abi_database_t
	var code C.int

	if config != nil {
		cName := C.CString(config.Name)
		defer C.free(unsafe.Pointer(cName))
		cConfig := C.abi_database_config_t{
			name:             cName,
			dimension:        C.size_t(config.Dimension),
			initial_capacity: C.size_t(config.InitialCapacity),
		}
		code = C.abi_database_create(&cConfig, &handle)
	} else {
		code = C.abi_database_create(nil, &handle)
	}

	if err := codeToError(code); err != nil {
		return nil, err
	}
	return &VectorDatabase{handle: handle}, nil
}

// Close releases all resources associated with the database.
func (db *VectorDatabase) Close() {
	if db.handle != nil {
		C.abi_database_close(db.handle)
		db.handle = nil
	}
}

// Insert adds a vector with the given ID and optional metadata to the database.
func (db *VectorDatabase) Insert(id uint64, vector []float32, metadata string) error {
	if db.handle == nil {
		return ErrNotInit
	}

	var cMeta *C.char
	if metadata != "" {
		cMeta = C.CString(metadata)
		defer C.free(unsafe.Pointer(cMeta))
	}

	code := C.abi_database_insert(
		db.handle,
		C.uint64_t(id),
		(*C.float)(unsafe.Pointer(&vector[0])),
		C.size_t(len(vector)),
		cMeta,
	)
	return codeToError(code)
}

// SearchResult represents a single search result.
type SearchResult struct {
	ID    uint64
	Score float32
}

// Search finds the k most similar vectors to the query.
func (db *VectorDatabase) Search(query []float32, k uint) ([]SearchResult, error) {
	if db.handle == nil {
		return nil, ErrNotInit
	}

	results := make([]C.abi_search_result_t, k)
	var count C.size_t

	code := C.abi_database_search(
		db.handle,
		(*C.float)(unsafe.Pointer(&query[0])),
		C.size_t(len(query)),
		C.size_t(k),
		&results[0],
		&count,
	)
	if err := codeToError(code); err != nil {
		return nil, err
	}

	out := make([]SearchResult, int(count))
	for i := 0; i < int(count); i++ {
		out[i] = SearchResult{
			ID:    uint64(results[i].id),
			Score: float32(results[i].score),
		}
	}
	return out, nil
}

// Delete removes a vector by ID.
func (db *VectorDatabase) Delete(id uint64) error {
	if db.handle == nil {
		return ErrNotInit
	}
	code := C.abi_database_delete(db.handle, C.uint64_t(id))
	return codeToError(code)
}

// Count returns the number of vectors in the database.
func (db *VectorDatabase) Count() (uint, error) {
	if db.handle == nil {
		return 0, ErrNotInit
	}
	var count C.size_t
	code := C.abi_database_count(db.handle, &count)
	if err := codeToError(code); err != nil {
		return 0, err
	}
	return uint(count), nil
}

// ============================================================================
// GPU
// ============================================================================

// GPU backend constants.
const (
	GPUBackendAuto   = 0
	GPUBackendCUDA   = 1
	GPUBackendVulkan = 2
	GPUBackendMetal  = 3
	GPUBackendWebGPU = 4
)

// GPUConfig configures a GPU context.
type GPUConfig struct {
	Backend         int
	DeviceIndex     int
	EnableProfiling bool
}

// GPU represents a GPU context.
type GPU struct {
	handle *C.abi_gpu_t
}

// GPUInit initializes a GPU context.
// Pass nil for auto-detection defaults.
func GPUInit(config *GPUConfig) (*GPU, error) {
	var handle *C.abi_gpu_t
	var code C.int

	if config != nil {
		cConfig := C.abi_gpu_config_t{
			backend:          C.int(config.Backend),
			device_index:     C.int(config.DeviceIndex),
			enable_profiling: C.bool(config.EnableProfiling),
		}
		code = C.abi_gpu_init(&cConfig, &handle)
	} else {
		code = C.abi_gpu_init(nil, &handle)
	}

	if err := codeToError(code); err != nil {
		return nil, err
	}
	return &GPU{handle: handle}, nil
}

// Shutdown releases GPU resources.
func (g *GPU) Shutdown() {
	if g.handle != nil {
		C.abi_gpu_shutdown(g.handle)
		g.handle = nil
	}
}

// GPUIsAvailable checks if any GPU backend is available.
func GPUIsAvailable() bool {
	return bool(C.abi_gpu_is_available())
}

// BackendName returns the active GPU backend name.
func (g *GPU) BackendName() string {
	return C.GoString(C.abi_gpu_backend_name(g.handle))
}

// ============================================================================
// Agent
// ============================================================================

// Agent backend constants.
const (
	AgentBackendEcho        = 0
	AgentBackendOpenAI      = 1
	AgentBackendOllama      = 2
	AgentBackendHuggingFace = 3
	AgentBackendLocal       = 4
)

// Agent status constants.
const (
	AgentStatusReady = 0
	AgentStatusBusy  = 1
	AgentStatusError = 2
)

// AgentConfig configures an AI agent.
type AgentConfig struct {
	Name          string
	Backend       int
	Model         string
	SystemPrompt  string
	Temperature   float32
	TopP          float32
	MaxTokens     uint32
	EnableHistory bool
}

// DefaultAgentConfig returns agent config with sensible defaults.
func DefaultAgentConfig() AgentConfig {
	return AgentConfig{
		Name:          "agent",
		Backend:       AgentBackendEcho,
		Model:         "gpt-4",
		Temperature:   0.7,
		TopP:          0.9,
		MaxTokens:     1024,
		EnableHistory: true,
	}
}

// AgentResponse holds the response from an agent send operation.
type AgentResponse struct {
	Text       string
	Length     uint
	TokensUsed uint64
}

// AgentStats holds agent conversation statistics.
type AgentStats struct {
	HistoryLength     uint
	UserMessages      uint
	AssistantMessages uint
	TotalCharacters   uint
	TotalTokensUsed   uint64
}

// Agent represents an AI agent.
type Agent struct {
	handle *C.abi_agent_t
}

// CreateAgent creates a new AI agent.
func CreateAgent(config *AgentConfig) (*Agent, error) {
	var handle *C.abi_agent_t
	var code C.int

	if config != nil {
		cName := C.CString(config.Name)
		defer C.free(unsafe.Pointer(cName))
		cModel := C.CString(config.Model)
		defer C.free(unsafe.Pointer(cModel))

		var cSystemPrompt *C.char
		if config.SystemPrompt != "" {
			cSystemPrompt = C.CString(config.SystemPrompt)
			defer C.free(unsafe.Pointer(cSystemPrompt))
		}

		cConfig := C.abi_agent_config_t{
			name:           cName,
			backend:        C.int(config.Backend),
			model:          cModel,
			system_prompt:  cSystemPrompt,
			temperature:    C.float(config.Temperature),
			top_p:          C.float(config.TopP),
			max_tokens:     C.uint32_t(config.MaxTokens),
			enable_history: C.bool(config.EnableHistory),
		}
		code = C.abi_agent_create(&cConfig, &handle)
	} else {
		code = C.abi_agent_create(nil, &handle)
	}

	if err := codeToError(code); err != nil {
		return nil, err
	}
	return &Agent{handle: handle}, nil
}

// Destroy releases all agent resources.
func (a *Agent) Destroy() {
	if a.handle != nil {
		C.abi_agent_destroy(a.handle)
		a.handle = nil
	}
}

// Send sends a message to the agent and returns the response.
func (a *Agent) Send(message string) (*AgentResponse, error) {
	if a.handle == nil {
		return nil, ErrNotInit
	}

	cMessage := C.CString(message)
	defer C.free(unsafe.Pointer(cMessage))

	var resp C.abi_agent_response_t
	code := C.abi_agent_send(a.handle, cMessage, &resp)
	if err := codeToError(code); err != nil {
		return nil, err
	}

	return &AgentResponse{
		Text:       C.GoString(resp.text),
		Length:     uint(resp.length),
		TokensUsed: uint64(resp.tokens_used),
	}, nil
}

// Status returns the current agent status.
func (a *Agent) Status() int {
	if a.handle == nil {
		return AgentStatusError
	}
	return int(C.abi_agent_get_status(a.handle))
}

// Stats returns agent conversation statistics.
func (a *Agent) Stats() (*AgentStats, error) {
	if a.handle == nil {
		return nil, ErrNotInit
	}

	var stats C.abi_agent_stats_t
	code := C.abi_agent_get_stats(a.handle, &stats)
	if err := codeToError(code); err != nil {
		return nil, err
	}

	return &AgentStats{
		HistoryLength:     uint(stats.history_length),
		UserMessages:      uint(stats.user_messages),
		AssistantMessages: uint(stats.assistant_messages),
		TotalCharacters:   uint(stats.total_characters),
		TotalTokensUsed:   uint64(stats.total_tokens_used),
	}, nil
}

// ClearHistory clears the agent's conversation history.
func (a *Agent) ClearHistory() error {
	if a.handle == nil {
		return ErrNotInit
	}
	code := C.abi_agent_clear_history(a.handle)
	return codeToError(code)
}

// SetTemperature sets the agent's temperature parameter.
func (a *Agent) SetTemperature(temperature float32) error {
	if a.handle == nil {
		return ErrNotInit
	}
	code := C.abi_agent_set_temperature(a.handle, C.float(temperature))
	return codeToError(code)
}

// SetMaxTokens sets the agent's max tokens parameter.
func (a *Agent) SetMaxTokens(maxTokens uint32) error {
	if a.handle == nil {
		return ErrNotInit
	}
	code := C.abi_agent_set_max_tokens(a.handle, C.uint32_t(maxTokens))
	return codeToError(code)
}

// Name returns the agent's name.
func (a *Agent) Name() string {
	if a.handle == nil {
		return "unknown"
	}
	return C.GoString(C.abi_agent_get_name(a.handle))
}
