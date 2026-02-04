//! AI Agent operations.

use crate::error::{check_error, Error, Result};
use crate::ffi::{self, AbiAgent, AbiAgentConfig};
use crate::Framework;
use std::ffi::{CStr, CString};
use std::ptr;

/// AI Agent for conversational interactions.
///
/// # Example
///
/// ```no_run
/// use abi::{Framework, Agent};
///
/// let framework = Framework::new().expect("Failed to init");
/// let agent = Agent::new(&framework, "assistant").expect("Failed to create agent");
///
/// let response = agent.chat("Hello, how are you?").expect("Failed to chat");
/// println!("Agent: {}", response);
/// ```
pub struct Agent {
    handle: AbiAgent,
}

impl Agent {
    /// Create a new AI agent with default settings.
    pub fn new(framework: &Framework, name: &str) -> Result<Self> {
        Self::with_config(framework, AgentConfig {
            name: name.to_string(),
            ..Default::default()
        })
    }

    /// Create a new AI agent with specific configuration.
    pub fn with_config(framework: &Framework, config: AgentConfig) -> Result<Self> {
        let c_name = CString::new(config.name.as_str())
            .map_err(|_| Error::InvalidArgument("Invalid name".into()))?;

        let c_persona = config.persona.as_ref().map(|p| CString::new(p.as_str()).ok()).flatten();

        let ffi_config = AbiAgentConfig {
            name: c_name.as_ptr(),
            persona: c_persona.as_ref().map(|p| p.as_ptr()).unwrap_or(ptr::null()),
            temperature: config.temperature,
            enable_history: config.enable_history,
        };

        let mut handle: AbiAgent = ptr::null_mut();
        let err = unsafe { ffi::abi_agent_create(framework.handle(), &ffi_config, &mut handle) };
        check_error(err)?;

        if handle.is_null() {
            return Err(Error::NullPointer);
        }

        Ok(Self { handle })
    }

    /// Send a message to the agent and get a response.
    ///
    /// # Arguments
    ///
    /// * `message` - The user's message
    ///
    /// # Returns
    ///
    /// The agent's response text.
    pub fn chat(&self, message: &str) -> Result<String> {
        let c_message = CString::new(message)
            .map_err(|_| Error::InvalidArgument("Invalid message".into()))?;

        let mut response_ptr: *mut libc::c_char = ptr::null_mut();
        let err = unsafe {
            ffi::abi_agent_chat(self.handle, c_message.as_ptr(), &mut response_ptr)
        };
        check_error(err)?;

        if response_ptr.is_null() {
            return Ok(String::new());
        }

        let response = unsafe {
            let cstr = CStr::from_ptr(response_ptr);
            let result = cstr.to_str().map_err(|_| Error::InvalidUtf8)?.to_string();
            ffi::abi_free_string(response_ptr);
            result
        };

        Ok(response)
    }

    /// Clear the conversation history.
    pub fn clear_history(&self) -> Result<()> {
        let err = unsafe { ffi::abi_agent_clear_history(self.handle) };
        check_error(err)
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

/// Agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Agent name.
    pub name: String,
    /// Persona name (e.g., "abbey", "abi"). None for default.
    pub persona: Option<String>,
    /// Sampling temperature (0.0-2.0). Higher = more creative.
    pub temperature: f32,
    /// Enable conversation history.
    pub enable_history: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "assistant".to_string(),
            persona: None,
            temperature: 0.7,
            enable_history: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.name, "assistant");
        assert!(config.persona.is_none());
        assert!((config.temperature - 0.7).abs() < 0.01);
        assert!(config.enable_history);
    }
}
