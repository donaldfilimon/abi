//! Trusted-caller training attestations; numerical training is not proved by metadata.
use super::{
    ContractError, Deserialize, Serialize, TeacherIdentity, TrainingProfile, digest, identity,
};
contract_struct!(ArtifactRef {
    version: u32,
    relative_path: String,
    sha256: String,
    byte_length: u64
});
/// Execution states; failure is terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunState {
    /// Inputs prepared.
    Prepared,
    /// Trainer executing.
    Running,
    /// Resumable interruption.
    Interrupted,
    /// Terminal failure.
    Failed,
    /// Artifacts and independent evaluation available.
    Completed,
}
/// Fixtures never qualify as production language models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelClassification {
    /// Caller attests actual language-model training.
    ProductionLlm,
    /// Synthetic fixture or demonstration.
    Fixture,
}
contract_struct!(RunInputs {
    profile: TrainingProfile,
    base: ArtifactRef,
    tokenizer: ArtifactRef,
    snapshot_id: String,
    seed: u64,
    configuration_digest: String,
    initial_tensor_digest: String,
    trainer_principal: String,
    classification: ModelClassification
});
contract_struct!(TensorChange {
    before_digest: String,
    after_digest: String,
    changed_tensors: u64
});
contract_struct!(RunReceipt { version: u32, run_id: String, inputs: RunInputs, state: RunState, step_count: u64, adapter: Option<ArtifactRef>, tensor_change: Option<TensorChange>, evaluation_id: Option<String>, provenance: String });
contract_struct!(EvaluationReceipt {
    version: u32,
    id: String,
    run_id: String,
    inputs: RunInputs,
    adapter: ArtifactRef,
    evaluator_principal: String,
    passed: bool,
    provenance: String
});
contract_struct!(ModelRegistration {
    version: u32,
    id: String,
    profile: TrainingProfile,
    completed_run_id: String,
    base: ArtifactRef,
    adapter: ArtifactRef,
    tokenizer: ArtifactRef,
    snapshot_id: String,
    evaluation_id: String,
    classification: ModelClassification,
    teacher: TeacherIdentity
});
fn invalid(reason: &str) -> ContractError {
    ContractError::InvalidRun(reason.into())
}
/// Enforces an exact single-component content-addressed filename.
pub fn validate_artifact(reference: &ArtifactRef) -> Result<(), ContractError> {
    digest(&reference.sha256, "artifact sha256")?;
    if reference.version != 1
        || reference.relative_path != format!("{}.artifact", reference.sha256)
        || reference.byte_length == 0
    {
        return Err(invalid("artifact schema, path or empty bytes"));
    }
    Ok(())
}
/// Structural checks only; caller attestation and filesystem evidence require a registry.
pub fn validate_run(receipt: &RunReceipt) -> Result<(), ContractError> {
    if receipt.version != 1 {
        return Err(invalid("run schema"));
    }
    identity(&receipt.run_id, "run id")?;
    identity(&receipt.provenance, "provenance")?;
    identity(&receipt.inputs.trainer_principal, "trainer principal")?;
    validate_artifact(&receipt.inputs.base)?;
    validate_artifact(&receipt.inputs.tokenizer)?;
    digest(&receipt.inputs.snapshot_id, "snapshot")?;
    digest(&receipt.inputs.configuration_digest, "configuration")?;
    digest(&receipt.inputs.initial_tensor_digest, "initial tensors")?;
    if let Some(adapter) = &receipt.adapter {
        validate_artifact(adapter)?;
    }
    if receipt.state == RunState::Completed {
        let change = receipt
            .tensor_change
            .as_ref()
            .ok_or_else(|| invalid("missing tensor change"))?;
        digest(&change.before_digest, "before tensor digest")?;
        digest(&change.after_digest, "after tensor digest")?;
        if change.changed_tensors == 0
            || change.before_digest != receipt.inputs.initial_tensor_digest
            || change.before_digest == change.after_digest
            || receipt.step_count == 0
            || receipt.adapter.is_none()
        {
            return Err(invalid("no actual tensor change or output"));
        }
        identity(
            receipt
                .evaluation_id
                .as_deref()
                .ok_or_else(|| invalid("missing evaluation"))?,
            "evaluation",
        )?;
    }
    Ok(())
}
/// Immutable inputs and monotonic progress; Failed/Completed never resume.
pub fn validate_run_transition(
    previous: &RunReceipt,
    next: &RunReceipt,
) -> Result<(), ContractError> {
    validate_run(previous)?;
    validate_run(next)?;
    if previous.run_id != next.run_id
        || previous.inputs != next.inputs
        || next.step_count < previous.step_count
    {
        return Err(invalid("changed run identity or regressed progress"));
    }
    let allowed = matches!(
        (previous.state, next.state),
        (
            RunState::Prepared | RunState::Interrupted,
            RunState::Running | RunState::Failed
        ) | (
            RunState::Running,
            RunState::Running | RunState::Interrupted | RunState::Failed | RunState::Completed
        )
    );
    if !allowed {
        return Err(invalid("illegal run transition"));
    }
    Ok(())
}
#[cfg(test)]
pub(super) mod tests {
    use super::*;
    pub(crate) fn prepared() -> RunReceipt {
        let artifact = ArtifactRef {
            version: 1,
            relative_path: format!("{}.artifact", "a".repeat(64)),
            sha256: "a".repeat(64),
            byte_length: 1,
        };
        RunReceipt {
            version: 1,
            run_id: "run".into(),
            inputs: RunInputs {
                profile: TrainingProfile::Abbey,
                base: artifact.clone(),
                tokenizer: artifact,
                snapshot_id: "b".repeat(64),
                seed: 1,
                configuration_digest: "c".repeat(64),
                initial_tensor_digest: "d".repeat(64),
                trainer_principal: "trainer".into(),
                classification: ModelClassification::Fixture,
            },
            state: RunState::Prepared,
            step_count: 0,
            adapter: None,
            tensor_change: None,
            evaluation_id: None,
            provenance: "test fixture".into(),
        }
    }
    #[test]
    fn changed_inputs_and_terminal_states_refuse() {
        let mut run = prepared();
        run.state = RunState::Interrupted;
        let mut next = run.clone();
        next.state = RunState::Running;
        assert!(validate_run_transition(&run, &next).is_ok());
        next.inputs.seed += 1;
        assert!(validate_run_transition(&run, &next).is_err());
        next.inputs = run.inputs.clone();
        run.state = RunState::Failed;
        assert!(validate_run_transition(&run, &next).is_err());
        next.state = RunState::Completed;
        assert!(validate_run_transition(&run, &next).is_err());
    }
}
