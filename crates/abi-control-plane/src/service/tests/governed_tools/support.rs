use abi_agent_host::local_approval::*;
use abi_capability::{lifecycle::*, *};
use std::{cell::RefCell, path::PathBuf};

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct State {
    preview: Option<LocalApprovalPreview>,
    decision: Option<ExactApproval>,
    intents: Vec<Digest>,
    terminal: Vec<(Digest, TerminalOutcome)>,
}
/// Test-only local authority; commits exact preview, decision and intent to scratch disk.
pub(super) struct Local {
    path: PathBuf,
    state: RefCell<State>,
}
impl Local {
    pub(super) fn new(path: PathBuf) -> Self {
        Self {
            path,
            state: RefCell::new(State::default()),
        }
    }
    fn save(&self) -> Result<(), LifecycleError> {
        let bytes =
            serde_json::to_vec(&*self.state.borrow()).map_err(|_| LifecycleError::InvalidState)?;
        std::fs::write(&self.path, bytes).map_err(|_| LifecycleError::Unavailable)
    }
    pub(super) fn preview(&self) -> LocalApprovalPreview {
        self.state.borrow().preview.clone().unwrap()
    }
}
impl LocalApprovalStore for Local {
    fn read_local_approval(
        &self,
        call_id: &str,
    ) -> Result<Option<LocalApprovalPreview>, LifecycleError> {
        Ok(self
            .state
            .borrow()
            .preview
            .as_ref()
            .filter(|p| p.request.call_id == call_id)
            .cloned())
    }
    fn request_local_approval(
        &self,
        p: &LocalApprovalPreview,
    ) -> Result<LocalApprovalStatus, LifecycleError> {
        let mut s = self.state.borrow_mut();
        if s.preview.as_ref().is_some_and(|old| old != p) {
            return Err(LifecycleError::InvalidState);
        }
        s.preview = Some(p.clone());
        let state = if s.decision.is_some() {
            LocalApprovalState::Approved
        } else {
            LocalApprovalState::Pending
        };
        drop(s);
        self.save()?;
        Ok(LocalApprovalStatus {
            call_id: p.request.call_id.clone(),
            preview_digest: p.digest(),
            state,
        })
    }
    fn decide_local_approval(
        &self,
        call: &str,
        digest: Digest,
        decision: &str,
        approved: bool,
        now: u64,
    ) -> Result<LocalApprovalStatus, LifecycleError> {
        let p = self.preview();
        if !approved || p.request.call_id != call || p.digest() != digest || now >= p.expires_at_ms
        {
            return Err(LifecycleError::InvalidState);
        }
        self.state.borrow_mut().decision = Some(ExactApproval {
            approval: Approval {
                decision_id: decision.into(),
                call_id: call.into(),
                call_digest: p.binding.request_digest,
                package_digest: p.binding.package_digest,
                grant_id: p.grant_id,
                approver: p.request.principal,
                coapprover: None,
                level: ApprovalLevel::A1Actor,
                expires_at_ms: p.expires_at_ms,
                state: ApprovalState::Approved,
            },
            binding: p.binding,
            issued_at_ms: now,
        });
        self.save()?;
        Ok(LocalApprovalStatus {
            call_id: call.into(),
            preview_digest: digest,
            state: LocalApprovalState::Approved,
        })
    }
    fn exact_local_approval(
        &self,
        call: &str,
        digest: Digest,
        now: u64,
    ) -> Result<Option<ExactApproval>, LifecycleError> {
        let p = self.preview();
        if p.request.call_id != call || p.digest() != digest || now >= p.expires_at_ms {
            return Err(LifecycleError::InvalidState);
        }
        Ok(self.state.borrow().decision.clone())
    }
}
impl DurableLifecycle for Local {
    fn refuse_replay(&mut self, id: Digest) -> Result<bool, LifecycleError> {
        Ok(self.state.borrow().intents.contains(&id))
    }
    fn append_stage(&mut self, _: &StageAudit) -> Result<(), LifecycleError> {
        self.save()
    }
    fn prepare(&mut self, r: &PrepareRequest) -> Result<PrepareOutcome, LifecycleError> {
        if !r.is_valid() || self.state.borrow().decision != r.approval {
            return Err(LifecycleError::InvalidState);
        }
        if self.state.borrow().intents.contains(&r.execution_id) {
            return Ok(PrepareOutcome::Uncertain);
        }
        self.state.borrow_mut().intents.push(r.execution_id);
        self.state.borrow_mut().decision = None;
        self.save()?;
        Ok(PrepareOutcome::Prepared(PreparedExecution::after_commit(
            r.execution_id,
            r.binding.clone(),
        )))
    }
    fn finish(&mut self, p: &PreparedExecution, o: &TerminalOutcome) -> Result<(), LifecycleError> {
        self.state
            .borrow_mut()
            .terminal
            .push((p.execution_id(), o.clone()));
        self.save()
    }
    fn lookup(&mut self, id: Digest) -> Result<Option<TerminalOutcome>, LifecycleError> {
        Ok(self
            .state
            .borrow()
            .terminal
            .iter()
            .find(|(key, _)| *key == id)
            .map(|(_, o)| o.clone()))
    }
}
