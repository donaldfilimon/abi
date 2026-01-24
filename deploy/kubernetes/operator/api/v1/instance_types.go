// =============================================================================
// ABI Instance Types
// =============================================================================
// This file contains the Go types for the AbiInstance custom resource.
// These types are used by the operator to reconcile AbiInstance resources.

package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specreplicaspath=.spec.replicas,statusreplicaspath=.status.readyReplicas
// +kubebuilder:printcolumn:name="Replicas",type="integer",JSONPath=".spec.replicas"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Version",type="string",JSONPath=".status.version"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// AbiInstance is the Schema for the abiinstances API
type AbiInstance struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   AbiInstanceSpec   `json:"spec,omitempty"`
	Status AbiInstanceStatus `json:"status,omitempty"`
}

// AbiInstanceSpec defines the desired state of AbiInstance
type AbiInstanceSpec struct {
	// Replicas is the number of ABI instance replicas
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	Replicas int32 `json:"replicas"`

	// Image configuration
	// +optional
	Image ImageSpec `json:"image,omitempty"`

	// Features configuration
	// +optional
	Features FeaturesSpec `json:"features,omitempty"`

	// GPU configuration
	// +optional
	GPU *GPUSpec `json:"gpu,omitempty"`

	// Resources defines the resource requirements
	// +optional
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`

	// Storage configuration
	// +optional
	Storage *StorageSpec `json:"storage,omitempty"`

	// Networking configuration
	// +optional
	Networking NetworkingSpec `json:"networking,omitempty"`

	// Config contains application configuration
	// +optional
	Config ConfigSpec `json:"config,omitempty"`

	// LLM configuration
	// +optional
	LLM *LLMSpec `json:"llm,omitempty"`

	// Secrets references
	// +optional
	Secrets *SecretsSpec `json:"secrets,omitempty"`

	// HighAvailability configuration
	// +optional
	HighAvailability *HASpec `json:"highAvailability,omitempty"`

	// Autoscaling configuration
	// +optional
	Autoscaling *AutoscalingSpec `json:"autoscaling,omitempty"`
}

// ImageSpec defines container image settings
type ImageSpec struct {
	// Repository is the container image repository
	// +kubebuilder:default="ghcr.io/abi/abi"
	Repository string `json:"repository,omitempty"`

	// Tag is the container image tag
	// +kubebuilder:default="latest"
	Tag string `json:"tag,omitempty"`

	// PullPolicy is the image pull policy
	// +kubebuilder:default="IfNotPresent"
	// +kubebuilder:validation:Enum=Always;IfNotPresent;Never
	PullPolicy corev1.PullPolicy `json:"pullPolicy,omitempty"`

	// PullSecrets are the image pull secrets
	// +optional
	PullSecrets []corev1.LocalObjectReference `json:"pullSecrets,omitempty"`
}

// FeaturesSpec defines which ABI features to enable
type FeaturesSpec struct {
	// AI enables AI/LLM features
	// +kubebuilder:default=true
	AI bool `json:"ai,omitempty"`

	// GPU enables GPU acceleration
	// +kubebuilder:default=false
	GPU bool `json:"gpu,omitempty"`

	// Database enables vector database
	// +kubebuilder:default=true
	Database bool `json:"database,omitempty"`

	// Network enables distributed networking
	// +kubebuilder:default=true
	Network bool `json:"network,omitempty"`

	// Web enables web/HTTP features
	// +kubebuilder:default=true
	Web bool `json:"web,omitempty"`

	// Profiling enables performance profiling
	// +kubebuilder:default=false
	Profiling bool `json:"profiling,omitempty"`
}

// GPUSpec defines GPU configuration
type GPUSpec struct {
	// Backend is the GPU backend to use
	// +kubebuilder:default="auto"
	// +kubebuilder:validation:Enum=auto;cuda;vulkan;metal;webgpu;none
	Backend string `json:"backend,omitempty"`

	// Count is the number of GPUs per replica
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=8
	Count int32 `json:"count,omitempty"`

	// Memory is the GPU memory limit (e.g., "8Gi")
	// +optional
	Memory string `json:"memory,omitempty"`
}

// StorageSpec defines persistent storage configuration
type StorageSpec struct {
	// Enabled enables persistent storage
	// +kubebuilder:default=true
	Enabled bool `json:"enabled,omitempty"`

	// Size is the storage size
	// +kubebuilder:default="10Gi"
	Size string `json:"size,omitempty"`

	// StorageClassName is the storage class name
	// +optional
	StorageClassName *string `json:"storageClassName,omitempty"`

	// AccessModes are the access modes
	// +kubebuilder:default={"ReadWriteOnce"}
	AccessModes []corev1.PersistentVolumeAccessMode `json:"accessModes,omitempty"`
}

// NetworkingSpec defines network configuration
type NetworkingSpec struct {
	// HTTPPort is the HTTP service port
	// +kubebuilder:default=8080
	HTTPPort int32 `json:"httpPort,omitempty"`

	// MetricsPort is the metrics port
	// +kubebuilder:default=9090
	MetricsPort int32 `json:"metricsPort,omitempty"`

	// GRPCPort is the gRPC service port
	// +kubebuilder:default=50051
	GRPCPort int32 `json:"grpcPort,omitempty"`

	// ServiceType is the Kubernetes service type
	// +kubebuilder:default="ClusterIP"
	// +kubebuilder:validation:Enum=ClusterIP;NodePort;LoadBalancer
	ServiceType corev1.ServiceType `json:"serviceType,omitempty"`
}

// ConfigSpec defines application configuration
type ConfigSpec struct {
	// LogLevel is the log level
	// +kubebuilder:default="info"
	// +kubebuilder:validation:Enum=debug;info;warn;error
	LogLevel string `json:"logLevel,omitempty"`

	// LogFormat is the log format
	// +kubebuilder:default="json"
	// +kubebuilder:validation:Enum=json;text
	LogFormat string `json:"logFormat,omitempty"`

	// Tracing enables distributed tracing
	// +kubebuilder:default=false
	Tracing bool `json:"tracing,omitempty"`

	// JaegerEndpoint is the Jaeger collector endpoint
	// +optional
	JaegerEndpoint string `json:"jaegerEndpoint,omitempty"`
}

// LLMSpec defines LLM provider configuration
type LLMSpec struct {
	// Provider is the LLM provider
	// +kubebuilder:default="ollama"
	// +kubebuilder:validation:Enum=ollama;openai;anthropic;huggingface;local
	Provider string `json:"provider,omitempty"`

	// OllamaHost is the Ollama service host
	// +kubebuilder:default="http://ollama-service:11434"
	OllamaHost string `json:"ollamaHost,omitempty"`

	// Model is the default model name
	// +kubebuilder:default="llama3"
	Model string `json:"model,omitempty"`
}

// SecretsSpec defines secret references
type SecretsSpec struct {
	// OpenAIAPIKeySecret references the OpenAI API key secret
	// +optional
	OpenAIAPIKeySecret *SecretKeyRef `json:"openaiApiKeySecret,omitempty"`

	// AnthropicAPIKeySecret references the Anthropic API key secret
	// +optional
	AnthropicAPIKeySecret *SecretKeyRef `json:"anthropicApiKeySecret,omitempty"`

	// HuggingFaceTokenSecret references the HuggingFace token secret
	// +optional
	HuggingFaceTokenSecret *SecretKeyRef `json:"huggingfaceTokenSecret,omitempty"`
}

// SecretKeyRef references a key in a secret
type SecretKeyRef struct {
	// Name is the secret name
	Name string `json:"name"`

	// Key is the key in the secret
	// +kubebuilder:default="api-key"
	Key string `json:"key,omitempty"`
}

// HASpec defines high availability configuration
type HASpec struct {
	// Enabled enables HA mode
	// +kubebuilder:default=false
	Enabled bool `json:"enabled,omitempty"`

	// PodDisruptionBudget configuration
	// +optional
	PodDisruptionBudget *PDBSpec `json:"podDisruptionBudget,omitempty"`

	// AntiAffinity is the pod anti-affinity mode
	// +kubebuilder:default="soft"
	// +kubebuilder:validation:Enum=soft;hard
	AntiAffinity string `json:"antiAffinity,omitempty"`
}

// PDBSpec defines PodDisruptionBudget configuration
type PDBSpec struct {
	// MinAvailable is the minimum number of available pods
	// +optional
	MinAvailable *int32 `json:"minAvailable,omitempty"`

	// MaxUnavailable is the maximum number of unavailable pods
	// +optional
	MaxUnavailable *int32 `json:"maxUnavailable,omitempty"`
}

// AutoscalingSpec defines HPA configuration
type AutoscalingSpec struct {
	// Enabled enables HPA
	// +kubebuilder:default=false
	Enabled bool `json:"enabled,omitempty"`

	// MinReplicas is the minimum number of replicas
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	MinReplicas int32 `json:"minReplicas,omitempty"`

	// MaxReplicas is the maximum number of replicas
	// +kubebuilder:default=10
	// +kubebuilder:validation:Minimum=1
	MaxReplicas int32 `json:"maxReplicas,omitempty"`

	// TargetCPUUtilization is the target CPU utilization percentage
	// +kubebuilder:default=80
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=100
	TargetCPUUtilization int32 `json:"targetCPUUtilization,omitempty"`

	// TargetMemoryUtilization is the target memory utilization percentage
	// +optional
	TargetMemoryUtilization *int32 `json:"targetMemoryUtilization,omitempty"`
}

// AbiInstanceStatus defines the observed state of AbiInstance
type AbiInstanceStatus struct {
	// Phase is the current phase of the ABI instance
	// +kubebuilder:validation:Enum=Pending;Creating;Running;Updating;Failed;Terminating
	Phase string `json:"phase,omitempty"`

	// Replicas is the total number of replicas
	Replicas int32 `json:"replicas,omitempty"`

	// ReadyReplicas is the number of ready replicas
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// AvailableReplicas is the number of available replicas
	AvailableReplicas int32 `json:"availableReplicas,omitempty"`

	// UpdatedReplicas is the number of updated replicas
	UpdatedReplicas int32 `json:"updatedReplicas,omitempty"`

	// Version is the current ABI version running
	Version string `json:"version,omitempty"`

	// ObservedGeneration is the most recent generation observed
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Health is the health status
	// +optional
	Health *HealthStatus `json:"health,omitempty"`

	// EnabledFeatures lists the enabled features
	// +optional
	EnabledFeatures []string `json:"enabledFeatures,omitempty"`

	// Conditions represent the latest available observations
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// Endpoints contains the service endpoints
	// +optional
	Endpoints *EndpointsStatus `json:"endpoints,omitempty"`

	// ConfigHash is the hash of the last applied configuration
	ConfigHash string `json:"configHash,omitempty"`
}

// HealthStatus represents the health of the instance
type HealthStatus struct {
	// Status is the health status
	// +kubebuilder:validation:Enum=Healthy;Degraded;Unhealthy;Unknown
	Status string `json:"status,omitempty"`

	// Message provides additional context
	Message string `json:"message,omitempty"`

	// LastChecked is when the health was last checked
	LastChecked metav1.Time `json:"lastChecked,omitempty"`
}

// EndpointsStatus contains service endpoint information
type EndpointsStatus struct {
	// HTTP is the HTTP endpoint
	HTTP string `json:"http,omitempty"`

	// GRPC is the gRPC endpoint
	GRPC string `json:"grpc,omitempty"`

	// Metrics is the metrics endpoint
	Metrics string `json:"metrics,omitempty"`
}

// +kubebuilder:object:root=true

// AbiInstanceList contains a list of AbiInstance
type AbiInstanceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AbiInstance `json:"items"`
}

func init() {
	SchemeBuilder.Register(&AbiInstance{}, &AbiInstanceList{})
}
