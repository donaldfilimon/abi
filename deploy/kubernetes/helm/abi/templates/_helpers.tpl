{{/*
Expand the name of the chart.
*/}}
{{- define "abi.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "abi.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "abi.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "abi.labels" -}}
helm.sh/chart: {{ include "abi.chart" . }}
{{ include "abi.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "abi.selectorLabels" -}}
app.kubernetes.io/name: {{ include "abi.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "abi.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "abi.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the image name
*/}}
{{- define "abi.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end }}

{{/*
Return the secret name for API keys
*/}}
{{- define "abi.secretName" -}}
{{- if .Values.secrets.create }}
{{- include "abi.fullname" . }}-secrets
{{- else }}
{{- .Values.secrets.existingSecret }}
{{- end }}
{{- end }}

{{/*
Return the configmap name
*/}}
{{- define "abi.configmapName" -}}
{{- include "abi.fullname" . }}-config
{{- end }}

{{/*
Return the PVC name
*/}}
{{- define "abi.pvcName" -}}
{{- include "abi.fullname" . }}-data
{{- end }}
