//! AI Agent operations.

use crate::error::{check_error, Error, Result};
use crate::ffi::{self, AbiAgent, AbiAgentConfig, AbiAgentResponse, AbiAgentStats};
use std::ffi::{CStr, CString};
use std::ptr;

/// AI Agent for conversational interactions.
///
/// # Example
///
/// ```no_run
/// use abi::Agent;
///
/// let agent = Agent::new("assistant").expect("Failed to create agent");
///
/// let response = agent.send("Hello, how are you?").expect("Failed to send");
/// println!("Agent: {} ({} tokens)", response.text, response.tokens_used);
/// ```
pub struct Agent {
    handle: AbiAgent,
}

impl Agent {
    /// Create a new AI agent with default settings.
    pub fn new(name: &str) -> Result<Self> {
        Self::with_config(AgentConfig {
            name: name.to_string(),
            ..Default::default()
        })
    }

    /// Create a new AI agent with specific configuration.
    pub fn with_config(config: AgentConfig) -> Result<Self> {
        let c_name = CString::new(config.name.as_str())
            .map_err(|_| Error::InvalidArgument("Invalid name".into()))?;
        let c_model = CString::new(config.model.as_str())
            .map_err(|_| Error::InvalidArgument("Invalid model".into()))?;
        let c_system_prompt = config
            .system_prompt
            .as_ref()
            .map(|p| CString::new(p.as_str()).ok())
            .flatten();

        let ffi_config = AbiAgentConfig {
            name: c_name.as_ptr(),
            backend: config.backend as libc::c_int,
            model: c_model.as_ptr(),
            system_prompt: c_system_prompt
                .as_ref()
                .map(|p| p.as_ptr())
                .unwrap_or(ptr::null()),
            temperature: config.temperature,
            top_p: config.top_p,
            max_tokens: config.max_tokens,
            enable_history: config.enable_history,
        };

        let mut handle: AbiAgent = ptr::null_mut();
        let err = unsafe { ffi::abi_agent_create(&ffi_config, &mut handle) };
        check_error(err)?;

        if handle.is_null() {
            return Err(Error::NullPointer);
        }

        Ok(Self { handle })
    }

    /// Send a message to the agent and get a response.
    pub fn send(&self, message: &str) -> Result<AgentResponse> {
        let c_message = CString::new(message)
            .map_err(|_| Error::InvalidArgument("Invalid message".into()))?;

        let mut response = AbiAgentResponse {
            text: ptr::null(),
            length: 0,
            tokens_used: 0,
        };

        let err =
            unsafe { ffi::abi_agent_send(self.handle, c_message.as_ptr(), &mut response) };
        check_error(err)?;

        let text = if response.text.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(response.text) }
                .to_str()
                .map_err(|_| Error::InvalidUtf8)?
                .to_string()
        };

        Ok(AgentResponse {
            text,
            length: response.length,
            tokens_used: response.tokens_used,
        })
    }

    /// Get the agent's current status code.
    pub fn status(&self) -> i32 {
        unsafe { ffi::abi_agent_get_status(self.handle) }
    }

    /// Get conversation statistics.
    pub fn stats(&self) -> Result<AgentStats> {
        let mut raw = AbiAgentStats {
            history_length: 0,
            user_messages: 0,
            assistant_messages: 0,
            total_characters: 0,
            total_tokens_used: 0,
        };

        let err = unsafe { ffi::abi_agent_get_stats(self.handle, &mut raw) };
        check_error(err)?;

        Ok(AgentStats {
            history_length: raw.history_length,
            user_messages: raw.user_messages,
            assistant_messages: raw.assistant_messages,
            total_characters: raw.total_characters,
            total_tokens_used: raw.total_tokens_used,
        })
    }

    /// Clear the conversation history.
    pub fn clear_history(&self) -> Result<()> {
        let err = unsafe { ffi::abi_agent_clear_history(self.handle) };
        check_error(err)
    }

    /// Set the sampling temperature.
    pub fn set_temperature(&self, temperature: f32) -> Result<()> {
        let err = unsafe { ffi::abi_agent_set_temperature(self.handle, temperature) };
        check_error(err)
    }

    /// Set the maximum generation tokens.
    pub fn set_max_tokens(&self, max_tokens: u32) -> Result<()> {
        let err = unsafe { ffi::abi_agent_set_max_tokens(self.handle, max_tokens) };
        check_error(err)
    }

    /// Get the agent's name.
    pub fn name(&self) -> &str {
        unsafe {
            let ptr = ffi::abi_agent_get_name(self.handle);
            if ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
            }
        }
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::abi_agent_destroy(self.handle);
            }
        }
    }
}

unsafe impl Send for Agent {}

/// Agent backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AgentBackend {
    Echo = 0,
    OpenAI = 1,
    Ollama = 2,
    HuggingFace = 3,
    Local = 4,
}

/// Agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Agent name.
    pub name: String,
    /// Backend type.
    pub backend: AgentBackend,
    /// Model name (e.g., "gpt-4").
    pub model: String,
    /// System prompt (optional).
    pub system_prompt: Option<String>,
    /// Sampling temperature (0.0-2.0). Higher = more creative.
    pub temperature: f32,
    /// Top-p nucleus sampling (0.0-1.0).
    pub top_p: f32,
    /// Maximum generation tokens.
    pub max_tokens: u32,
    /// Enable conversation history.
    pub enable_history: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "assistant".to_string(),
            backend: AgentBackend::Echo,
            model: String::new(),
            system_prompt: None,
            temperature: 0.7,
            top_p: 1.0,
            max_tokens: 2048,
            enable_history: true,
        }
    }
}

/// Response from an agent send operation.
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// Response text.
    pub text: String,
    /// Length of response text.
    pub length: usize,
    /// Number of tokens used.
    pub tokens_used: u64,
}

/// Agent conversation statistics.
#[derive(Debug, Clone)]
pub struct AgentStats {
    /// Total messages in history.
    pub history_length: usize,
    /// User messages sent.
    pub user_messages: usize,
    /// Assistant messages received.
    pub assistant_messages: usize,
    /// Total characters exchanged.
    pub total_characters: usize,
    /// Total tokens used.
    pub total_tokens_used: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.name, "assistant");
        assert_eq!(config.backend, AgentBackend::Echo);
        assert!(config.model.is_empty());
        assert!(config.system_prompt.is_none());
        assert!((config.temperature - 0.7).abs() < 0.01);
        assert!((config.top_p - 1.0).abs() < 0.01);
        assert_eq!(config.max_tokens, 2048);
        assert!(config.enable_history);
    }

    #[test]
    fn test_agent_backend_values() {
        assert_eq!(AgentBackend::Echo as i32, 0);
        assert_eq!(AgentBackend::OpenAI as i32, 1);
        assert_eq!(AgentBackend::Ollama as i32, 2);
        assert_eq!(AgentBackend::HuggingFace as i32, 3);
        assert_eq!(AgentBackend::Local as i32, 4);
    }
}
