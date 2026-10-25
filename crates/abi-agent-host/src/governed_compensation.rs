//! Explicit inverse submission over the same governance path.
use super::*;
impl GovernedHost<'_> {
    /// Explicit compensation submission. Only a known failed original intent admits
    /// its declared inverse, with the inverse's own fresh grant/approval and scope.
    #[allow(clippy::too_many_arguments)]
    pub fn run_compensation(
        &self,
        original_tool: &str,
        original: &PreparedExecution,
        request: &ModelRequest,
        sink: &mut dyn EventSink,
        cancellation: &CancellationToken,
        options: GovernedOptions<'_>,
        facts: &mut dyn GovernedFactProvider,
        ledger: &mut dyn DurableLifecycle,
        actuator: &mut RecordingActuator,
    ) -> Result<GovernedRunReport, GovernedCompensationError> {
        let refuse = || GovernedCompensationError::InvalidOriginal;
        let registration = self.registrations.get(original_tool).ok_or_else(refuse)?;
        let source = self
            .sources
            .get(
                &registration.request.capability_id,
                &registration.request.capability_version,
            )
            .ok_or_else(refuse)?;
        if original.binding().package_digest != source.package().digest
            || original.binding().scope_digest != commitment(&registration.request.scope)
            || !matches!(
                ledger
                    .lookup(original.execution_id())
                    .map_err(GovernedCompensationError::Lifecycle)?,
                Some(TerminalOutcome {
                    state: OutcomeState::Failed,
                    ..
                })
            )
        {
            return Err(refuse());
        }
        let inverse = source.source().compensation.as_ref().ok_or_else(refuse)?;
        let registrations: Vec<_> = self
            .registrations
            .values()
            .filter(|candidate| {
                candidate.request.capability_id == inverse.capability_id
                    && candidate.request.capability_version == inverse.version
                    && candidate.request.scope == registration.request.scope
            })
            .cloned()
            .collect();
        if registrations.is_empty() {
            return Err(refuse());
        }
        // Never inherit the original approval. Canonical matching requires a new
        // exact inverse preview (or its own A0 grant), and a unique call identity.
        let inverse_host = Self::new(self.provider, self.sources, registrations)
            .map_err(GovernedCompensationError::Lifecycle)?;
        inverse_host
            .run_mode(
                request,
                sink,
                cancellation,
                options,
                facts,
                ledger,
                dispatch::Dispatch::Recording(actuator),
                Some(CompensationContext {
                    forward: source,
                    original,
                }),
            )
            .map_err(GovernedCompensationError::Driver)
    }
}
