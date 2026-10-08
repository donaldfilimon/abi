//! Run/model mutations share the registry's owned writer, budget and poison rules.
use super::{
    BTreeMap, Deserialize, Eligibility, QualificationPurpose, RegistryError, Serialize,
    TrainingRegistry, insert, nonblank,
};
use crate::training_artifacts::verify_artifact;
use abi_ai::training_contracts::{
    ArtifactRef, EvaluationReceipt, ModelClassification, ModelRegistration, RunReceipt, RunState,
    TeacherIdentity, validate_artifact, validate_run, validate_run_transition,
};
use std::path::Path;
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Runs {
    artifacts: BTreeMap<String, ArtifactRef>,
    history: BTreeMap<String, Vec<RunReceipt>>,
    evaluations: BTreeMap<String, EvaluationReceipt>,
    models: BTreeMap<String, ModelRegistration>,
}
/// Selected identities with current eligibility. Service cutover still needs an operator.
#[derive(Debug, Clone)]
pub struct QualifiedModel {
    /// Verified registered model identities.
    pub registration: ModelRegistration,
    /// Current snapshot eligibility, invalidated by any registry mutation.
    pub eligibility: Eligibility,
    /// Always true; this library never performs service cutover.
    pub operator_cutover_required: bool,
}
impl TrainingRegistry {
    pub(crate) fn artifact_session_healthy(&self) -> Result<(), RegistryError> {
        self.healthy()
    }
    pub(crate) fn register_artifact(&mut self, artifact: ArtifactRef) -> Result<(), RegistryError> {
        self.healthy()?;
        validate_artifact(&artifact)?;
        let mut next = self.state.clone();
        if insert(&mut next.runs.artifacts, artifact.sha256.clone(), artifact)? {
            self.publish(next)?;
        }
        Ok(())
    }
    fn artifact(&self, artifact: &ArtifactRef) -> Result<(), RegistryError> {
        validate_artifact(artifact)?;
        if self.state.runs.artifacts.get(&artifact.sha256) != Some(artifact) {
            return Err(RegistryError::Missing("published artifact".into()));
        }
        Ok(())
    }
    fn physical(&self, root: &Path, artifact: &ArtifactRef) -> Result<(), RegistryError> {
        self.artifact(artifact)?;
        verify_artifact(root, artifact).map_err(|e| RegistryError::Invalid(e.to_string()))
    }
    fn run_structure(&self, run: &RunReceipt) -> Result<(), RegistryError> {
        validate_run(run)?;
        self.principal(&run.inputs.trainer_principal)?;
        self.artifact(&run.inputs.base)?;
        self.artifact(&run.inputs.tokenizer)?;
        let snapshot = self
            .state
            .snapshots
            .get(&run.inputs.snapshot_id)
            .ok_or_else(|| RegistryError::Missing("run snapshot".into()))?;
        if snapshot
            .records
            .iter()
            .any(|r| r.profile != run.inputs.profile)
        {
            return Err(RegistryError::Invalid("run profile mismatch".into()));
        }
        super::train_only(snapshot)?;
        if let Some(adapter) = &run.adapter {
            self.artifact(adapter)?;
        }
        if run.state == RunState::Completed {
            self.completed_evaluation(run)?;
        }
        Ok(())
    }
    fn completed_evaluation(&self, run: &RunReceipt) -> Result<(), RegistryError> {
        let id = run
            .evaluation_id
            .as_ref()
            .ok_or_else(|| RegistryError::Missing("evaluation".into()))?;
        let evaluation = self
            .state
            .runs
            .evaluations
            .get(id)
            .ok_or_else(|| RegistryError::Missing(id.clone()))?;
        self.evaluation_structure(evaluation)?;
        if evaluation.run_id != run.run_id
            || evaluation.inputs != run.inputs
            || Some(&evaluation.adapter) != run.adapter.as_ref()
        {
            return Err(RegistryError::Invalid("evaluation subject mismatch".into()));
        }
        Ok(())
    }
    fn evaluation_structure(&self, evaluation: &EvaluationReceipt) -> Result<(), RegistryError> {
        nonblank(&evaluation.id)?;
        nonblank(&evaluation.provenance)?;
        self.authorize(
            &evaluation.evaluator_principal,
            &self.policy.qualification_reviewers,
        )?;
        if evaluation.version != 1
            || !evaluation.passed
            || self.principal(&evaluation.evaluator_principal)?
                == self.principal(&evaluation.inputs.trainer_principal)?
        {
            return Err(RegistryError::Unauthorized);
        }
        self.artifact(&evaluation.adapter)?;
        let run = self
            .state
            .runs
            .history
            .get(&evaluation.run_id)
            .and_then(|h| h.last())
            .ok_or_else(|| RegistryError::Missing("evaluated run".into()))?;
        if run.inputs != evaluation.inputs {
            return Err(RegistryError::Invalid("evaluation inputs mismatch".into()));
        }
        Ok(())
    }
    /// Appends a state transition after checking current eligibility and physical
    /// inputs. Caller authenticates the registrar; receipts attest numerical work.
    pub fn register_run(
        &mut self,
        caller: &str,
        root: &Path,
        eligibility: &Eligibility,
        run: RunReceipt,
    ) -> Result<(), RegistryError> {
        self.healthy()?;
        self.authorize(caller, &self.policy.evidence_registrars)?;
        self.validate_eligibility(eligibility)?;
        if eligibility.snapshot_id != run.inputs.snapshot_id {
            return Err(RegistryError::Invalid(
                "eligibility subject mismatch".into(),
            ));
        }
        self.run_structure(&run)?;
        self.physical(root, &run.inputs.base)?;
        self.physical(root, &run.inputs.tokenizer)?;
        if let Some(adapter) = &run.adapter {
            self.physical(root, adapter)?;
        }
        if let Some(previous) = self
            .state
            .runs
            .history
            .get(&run.run_id)
            .and_then(|h| h.last())
        {
            validate_run_transition(previous, &run)?;
        } else if run.state != RunState::Prepared
            || run.step_count != 0
            || run.adapter.is_some()
            || run.tensor_change.is_some()
            || run.evaluation_id.is_some()
        {
            return Err(RegistryError::Invalid(
                "new run must start Prepared without output".into(),
            ));
        }
        let mut next = self.state.clone();
        next.runs
            .history
            .entry(run.run_id.clone())
            .or_default()
            .push(run);
        self.publish(next)
    }
    /// Registers an immutable independent passing evaluation of an exact run/output.
    pub fn register_evaluation(
        &mut self,
        caller: &str,
        root: &Path,
        evaluation: EvaluationReceipt,
    ) -> Result<(), RegistryError> {
        self.healthy()?;
        self.authorize(caller, &self.policy.evidence_registrars)?;
        self.evaluation_structure(&evaluation)?;
        self.eligibility(&evaluation.inputs.snapshot_id)?;
        self.physical(root, &evaluation.inputs.base)?;
        self.physical(root, &evaluation.inputs.tokenizer)?;
        self.physical(root, &evaluation.adapter)?;
        let run = self
            .state
            .runs
            .history
            .get(&evaluation.run_id)
            .and_then(|h| h.last())
            .ok_or_else(|| RegistryError::Missing("run".into()))?;
        if run.state != RunState::Running {
            return Err(RegistryError::Invalid(
                "evaluation requires running run".into(),
            ));
        }
        let mut next = self.state.clone();
        if insert(
            &mut next.runs.evaluations,
            evaluation.id.clone(),
            evaluation,
        )? {
            self.publish(next)?;
        }
        Ok(())
    }
    fn model_structure(&self, model: &ModelRegistration) -> Result<(), RegistryError> {
        nonblank(&model.id)?;
        let run = self
            .state
            .runs
            .history
            .get(&model.completed_run_id)
            .and_then(|h| h.last())
            .ok_or_else(|| RegistryError::Missing("completed run".into()))?;
        self.run_structure(run)?;
        if model.version != 1
            || model.classification != ModelClassification::ProductionLlm
            || run.inputs.classification != model.classification
            || run.state != RunState::Completed
            || model.profile != run.inputs.profile
            || model.base != run.inputs.base
            || model.tokenizer != run.inputs.tokenizer
            || Some(&model.adapter) != run.adapter.as_ref()
            || model.snapshot_id != run.inputs.snapshot_id
            || Some(&model.evaluation_id) != run.evaluation_id.as_ref()
        {
            return Err(RegistryError::Invalid(
                "model/run mismatch or fixture".into(),
            ));
        }
        let teacher = TeacherIdentity {
            base_digest: model.base.sha256.clone(),
            adapter_digest: model.adapter.sha256.clone(),
            tokenizer_digest: model.tokenizer.sha256.clone(),
            generation_configuration_digest: model.teacher.generation_configuration_digest.clone(),
            qualification_digest: model.teacher.qualification_digest.clone(),
        };
        let q = self.qualification(&teacher.qualification_digest, QualificationPurpose::Teacher)?;
        if model.teacher != teacher
            || q.artifact.base_digest != teacher.base_digest
            || q.artifact.adapter_digest != teacher.adapter_digest
            || q.artifact.tokenizer_digest != teacher.tokenizer_digest
            || q.artifact.generation_configuration_digest != teacher.generation_configuration_digest
            || self.principal(&q.artifact_principal)?
                != self.principal(&run.inputs.trainer_principal)?
        {
            return Err(RegistryError::Invalid(
                "teacher qualification mismatch".into(),
            ));
        }
        let eval = self
            .state
            .runs
            .evaluations
            .get(&model.evaluation_id)
            .ok_or_else(|| RegistryError::Missing("evaluation".into()))?;
        if self.principal(&eval.evaluator_principal)? == self.principal(&q.artifact_principal)? {
            return Err(RegistryError::Unauthorized);
        }
        Ok(())
    }
    /// Registers only physically verified production models with exact completed evidence.
    pub fn register_model(
        &mut self,
        caller: &str,
        root: &Path,
        model: ModelRegistration,
    ) -> Result<(), RegistryError> {
        self.healthy()?;
        self.authorize(caller, &self.policy.evidence_registrars)?;
        self.verify_model(root, &model)?;
        let mut next = self.state.clone();
        if insert(&mut next.runs.models, model.id.clone(), model)? {
            self.publish(next)?;
        }
        Ok(())
    }
    fn verify_model(
        &self,
        root: &Path,
        model: &ModelRegistration,
    ) -> Result<Eligibility, RegistryError> {
        self.healthy()?;
        self.model_structure(model)?;
        let eligibility = self.eligibility(&model.snapshot_id)?;
        for artifact in [&model.base, &model.adapter, &model.tokenizer] {
            self.physical(root, artifact)?;
        }
        Ok(eligibility)
    }
    pub(super) fn validate_runs(&self) -> Result<(), RegistryError> {
        for (digest, artifact) in &self.state.runs.artifacts {
            validate_artifact(artifact)?;
            if digest != &artifact.sha256 {
                return Err(RegistryError::Invalid("artifact key mismatch".into()));
            }
        }
        for (id, history) in &self.state.runs.history {
            let first = history
                .first()
                .ok_or_else(|| RegistryError::Invalid("empty run history".into()))?;
            if first.state != RunState::Prepared
                || first.step_count != 0
                || first.adapter.is_some()
                || first.tensor_change.is_some()
                || first.evaluation_id.is_some()
            {
                return Err(RegistryError::Invalid("invalid initial run".into()));
            }
            for run in history {
                if id != &run.run_id {
                    return Err(RegistryError::Invalid("run key mismatch".into()));
                }
                self.run_structure(run)?;
            }
            for pair in history.windows(2) {
                validate_run_transition(&pair[0], &pair[1])?;
            }
        }
        for (id, evaluation) in &self.state.runs.evaluations {
            if id != &evaluation.id {
                return Err(RegistryError::Invalid("evaluation key mismatch".into()));
            }
            self.evaluation_structure(evaluation)?;
        }
        for (id, model) in &self.state.runs.models {
            if id != &model.id {
                return Err(RegistryError::Invalid("model key mismatch".into()));
            }
            self.model_structure(model)?;
        }
        Ok(())
    }
}
/// Resolves a previously registered exact model, current revocations, independent
/// evaluation and physical bytes. No provider/service operation is performed.
pub fn select_model(
    registration: &ModelRegistration,
    registry: &TrainingRegistry,
    root: &Path,
) -> Result<QualifiedModel, RegistryError> {
    registry.healthy()?;
    if registry.state.runs.models.get(&registration.id) != Some(registration) {
        return Err(RegistryError::Missing("registered model".into()));
    }
    let eligibility = registry.verify_model(root, registration)?;
    Ok(QualifiedModel {
        registration: registration.clone(),
        eligibility,
        operator_cutover_required: true,
    })
}
