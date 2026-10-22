//! The one-line-per-field completion metadata block.
//!
//! Split out of `complete.rs` so the metadata shape and its rendering live
//! apart from argument parsing and the dispatch paths. The field names and
//! their order are part of the observable CLI contract.

use std::fmt::Write as _;

use abi_wdbx::RecordId;

/// Fields for the one-line-per-field completion metadata block.
pub(crate) struct MetaReport<'a> {
    pub(crate) requested_model: &'a str,
    pub(crate) profile: &'a str,
    pub(crate) audit_passed: bool,
    pub(crate) escore: f32,
    pub(crate) audit_vetoed: bool,
    pub(crate) persisted: bool,
    /// `(evidence_count, adapted, self_learn_cycle)` for a learning run.
    pub(crate) learn: Option<(usize, bool, &'a str)>,
    pub(crate) kv: usize,
    pub(crate) vectors: usize,
    pub(crate) blocks: usize,
    pub(crate) query_id: Option<RecordId>,
    pub(crate) response_id: Option<RecordId>,
    pub(crate) block_hex: Option<&'a str>,
    pub(crate) wdbx_status: Option<&'a str>,
    pub(crate) output: &'a str,
}

/// Render the local-completion metadata block.
///
/// The `self_learn_cycle` field states what actually happened: the full
/// persistence+adaptation+recall cycle only completes when the learning write
/// was committed. A skipped or unknown write must never be reported as a
/// completed cycle.
///
/// Every metadata field lives on the FIRST output line. The checked-in SEA
/// learn smoke (`.agents/skills/sea-learn-loop/learn.sh` and its `.claude`
/// mirror) inspects only the first line on purpose: the generated response
/// text is free to repeat arbitrary markers, so a later line could otherwise
/// make a failed learning write look healthy. Moving the learn fields onto
/// their own line hid `provider=local`/`persisted=` from that check and
/// failed a healthy run with "missing metadata: provider=local". Keep the
/// learn fields on the first line with the rest of the metadata.
pub(crate) fn render_local(report: &MetaReport<'_>) -> String {
    let mut out = String::new();
    let learn_fields = match report.learn {
        Some((evidence_count, adapted, cycle)) => format!(
            " learn=true evidence_count={evidence_count} adapted={adapted} self_learn_cycle={cycle} recall_ready={}",
            if evidence_count > 0 {
                "yes"
            } else {
                "first-pass"
            }
        ),
        None => String::new(),
    };
    let _ = writeln!(
        out,
        "requested_model={} provider=local transport=in-process generation_engine=persona-template policy_scope=lexical-signal profile={} audit_passed={} audit_escore={:.3} audit_vetoed={} persisted={}{learn_fields}",
        report.requested_model,
        report.profile,
        report.audit_passed,
        report.escore,
        report.audit_vetoed,
        report.persisted,
    );
    let _ = writeln!(
        out,
        "wdbx kv_entries={} vectors={} blocks={}",
        report.kv, report.vectors, report.blocks
    );
    if let Some(qid) = report.query_id {
        let _ = writeln!(out, "query_vector_id={qid}");
        let _ = writeln!(out, "metadata_key=completion:{qid}");
    }
    if let Some(rid) = report.response_id {
        let _ = writeln!(out, "response_vector_id={rid}");
    }
    if let Some(hex) = report.block_hex {
        let _ = writeln!(out, "block_id={hex}");
    }
    if let Some(status) = report.wdbx_status {
        let _ = writeln!(out, "wdbx_status={status}");
    }
    let _ = writeln!(out, "{}", report.output);
    out
}
