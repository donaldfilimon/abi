//! Rust port of `tools/ci_contract.py`.
//! Mirrors Python logic exactly for byte-identical oracle parity.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use regex::Regex;

/// Return parent-sibling checkout roots and their manifest paths.
/// Mirrors `sibling_dependency_requirements` in Python.
#[must_use]
pub fn sibling_dependency_requirements(cargo_toml: &str) -> BTreeMap<String, Vec<String>> {
    let value: toml::Value = match cargo_toml.parse() {
        Ok(v) => v,
        Err(_) => return BTreeMap::new(),
    };
    let deps = value
        .get("workspace")
        .and_then(|w| w.get("dependencies"))
        .and_then(|d| d.as_table());
    let Some(deps) = deps else {
        return BTreeMap::new();
    };
    let mut requirements: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for spec in deps.values() {
        let Some(table) = spec.as_table() else {
            continue;
        };
        let Some(path_val) = table.get("path").and_then(|v| v.as_str()) else {
            continue;
        };
        // PurePosixPath logic: split on '/'
        let parts: Vec<&str> = path_val.split('/').collect();
        if parts.len() < 3 || parts[0] != ".." {
            continue;
        }
        let sibling = parts[1].to_string();
        requirements
            .entry(sibling)
            .or_default()
            .push(path_val.to_string());
    }
    // sort inner vectors and return sorted map
    for paths in requirements.values_mut() {
        paths.sort();
    }
    requirements
}

/// Extract repository owner from Cargo workspace package repository.
/// Mirrors `_repository_owner`.
pub fn repository_owner(cargo_toml: &str) -> Result<String, String> {
    let value: toml::Value = cargo_toml
        .parse()
        .map_err(|e| format!("workspace repository must be a GitHub URL: {e}"))?;
    let repository = value
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("repository"))
        .and_then(|r| r.as_str())
        .ok_or_else(|| "workspace repository must be a GitHub URL".to_string())?;
    let re = Regex::new(r"^https://github\.com/([^/]+)/[^/]+$").unwrap();
    let caps = re
        .captures(repository)
        .ok_or_else(|| "workspace repository must be a GitHub URL".to_string())?;
    Ok(caps.get(1).unwrap().as_str().to_string())
}

fn has_yaml_indirection(line: &str) -> bool {
    // Mirrors Python: (?<![\w])[&*](?![&*])[^\s\[\]{},'"&*]+
    let chars: Vec<char> = line.chars().collect();
    let len = chars.len();
    for i in 0..len {
        let c = chars[i];
        if c != '&' && c != '*' {
            continue;
        }
        // (?<![\w])
        if i > 0 {
            let prev = chars[i - 1];
            if prev.is_ascii_alphanumeric() || prev == '_' {
                continue;
            }
        }
        // (?![&*])
        if i + 1 < len && (chars[i + 1] == '&' || chars[i + 1] == '*') {
            continue;
        }
        if i + 1 >= len {
            continue;
        }
        // [^\s\[\]{},'"&*]+ at least one
        let mut j = i + 1;
        let mut count = 0;
        while j < len {
            let ch = chars[j];
            if ch.is_ascii_whitespace()
                || matches!(ch, '[' | ']' | '{' | '}' | ',' | '\'' | '"' | '&' | '*')
            {
                break;
            }
            count += 1;
            j += 1;
        }
        if count > 0 {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone)]
struct CheckoutStep {
    indent: usize,
    block: Vec<String>,
}

fn checkout_steps(workflow: &str) -> Vec<CheckoutStep> {
    let lines: Vec<String> = workflow.lines().map(ToString::to_string).collect();
    let re_item = Regex::new(r"^(\s*)-(?:\s+.*)?$").unwrap();
    let re_checkout = Regex::new(r"(?i)actions/checkout@").unwrap();
    let mut steps: Vec<CheckoutStep> = Vec::new();
    let mut index = 0usize;
    while index < lines.len() {
        let Some(caps) = re_item.captures(&lines[index]) else {
            index += 1;
            continue;
        };
        let indent = caps.get(1).unwrap().as_str().len();
        let mut end = index + 1;
        while end < lines.len() {
            let candidate = &lines[end];
            if !candidate.trim().is_empty() {
                let candidate_indent = candidate.len() - candidate.trim_start().len();
                if candidate_indent < indent {
                    break;
                }
                if candidate_indent == indent && re_item.is_match(candidate) {
                    break;
                }
            }
            end += 1;
        }
        let block: Vec<String> = lines[index..end].to_vec();
        let has_checkout = block.iter().any(|line| {
            if line.trim_start().starts_with('#') {
                return false;
            }
            re_checkout.is_match(line)
        });
        if has_checkout {
            steps.push(CheckoutStep { indent, block });
        }
        index = std::cmp::max(end, index + 1);
    }
    steps
}

fn checkout_disables_persisted_credentials(indent: usize, step: &[String]) -> bool {
    let with_re = Regex::new(&format!(r"^\s{{{}}}with:\s*(?:#.*)?$", indent + 2)).unwrap();
    let value_re = Regex::new(&format!(
        r"^\s{{{}}}persist-credentials:\s*([^#]+?)(?:\s+#.*)?$",
        indent + 4
    ))
    .unwrap();
    let mut in_with = false;
    for line in step {
        let line_indent = line.len() - line.trim_start().len();
        if with_re.is_match(line) {
            in_with = true;
            continue;
        }
        if !in_with {
            continue;
        }
        if !line.trim().is_empty() && line_indent <= indent + 2 {
            break;
        }
        if let Some(caps) = value_re.captures(line) {
            let raw = caps.get(1).unwrap().as_str().trim();
            if matches!(raw, "false" | "\"false\"" | "'false'") {
                return true;
            }
            return false;
        }
    }
    false
}

/// Return repository-wide checkout credential policy failures.
/// Mirrors `validate_checkout_credentials`.
#[must_use]
pub fn validate_checkout_credentials(workflow: &str) -> Vec<String> {
    let failure = "every checkout must disable persisted credentials".to_string();
    let uses_key = Regex::new(r#"(?i)(?:^|[-{,]\s*)(?:["']uses["']|uses)\s*:"#).unwrap();
    let block_scalar = Regex::new(r":\s*[>|][-+]?\s*$").unwrap();

    for line in workflow.lines() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        if has_yaml_indirection(line) || line.contains('\\') {
            return vec![failure.clone()];
        }
        if uses_key.is_match(line) && block_scalar.is_match(line) {
            return vec![failure.clone()];
        }
    }

    let steps = checkout_steps(workflow);
    let re_checkout = Regex::new(r"(?i)actions/checkout@").unwrap();
    let checkout_mentions: usize = workflow
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .map(|l| re_checkout.find_iter(l).count())
        .sum();
    if checkout_mentions != steps.len() {
        return vec![failure];
    }
    for step in &steps {
        if !checkout_disables_persisted_credentials(step.indent, &step.block) {
            return vec![failure];
        }
    }
    Vec::new()
}

fn job_sections(workflow: &str) -> BTreeMap<String, String> {
    let marker = "\njobs:\n";
    let Some(pos) = workflow.find(marker) else {
        return BTreeMap::new();
    };
    let body = &workflow[pos + marker.len()..];
    let re = Regex::new(r"(?m)^  ([a-zA-Z0-9_-]+):\s*$").unwrap();
    let matches: Vec<(usize, usize, String)> = re
        .captures_iter(body)
        .map(|c| {
            let m = c.get(0).unwrap();
            let name = c.get(1).unwrap().as_str().to_string();
            (m.start(), m.end(), name)
        })
        .collect();
    let mut sections: BTreeMap<String, String> = BTreeMap::new();
    for (idx, (start, _, name)) in matches.iter().enumerate() {
        let end = if idx + 1 < matches.len() {
            matches[idx + 1].0
        } else {
            body.len()
        };
        let section = body[*start..end].to_string();
        sections.insert(name.clone(), section);
    }
    sections
}

/// Validate workflow safety failures, deduped preserving order.
/// Mirrors `validate_workflow`.
pub fn validate_workflow(workflow: &str, cargo_toml: &str) -> Vec<String> {
    let siblings = sibling_dependency_requirements(cargo_toml);
    let owner = match repository_owner(cargo_toml) {
        Ok(o) => o,
        Err(e) => return vec![e],
    };
    let mut failures: Vec<String> = Vec::new();

    let full = Regex::new(r"^[0-9a-f]{40}$").unwrap();
    for sibling in siblings.keys() {
        let env_name = format!("{}_REVISION", sibling.to_uppercase().replace('-', "_"));
        let pattern = format!(r"(?m)^  {}:\s*([^\s#]+)", regex::escape(&env_name));
        let re = Regex::new(&pattern).unwrap();
        let Some(caps) = re.captures(workflow) else {
            failures.push(format!("{sibling} revision must be an immutable commit"));
            continue;
        };
        let rev = caps.get(1).unwrap().as_str();
        if !full.is_match(rev) {
            failures.push(format!("{sibling} revision must be an immutable commit"));
        }
    }

    if workflow.contains("WDBX_CHECKOUT_TOKEN")
        || Regex::new(r"(?m)^\s*token:\s*\$\{\{\s*secrets\.")
            .unwrap()
            .is_match(workflow)
    {
        failures.push("wdbx checkout must not use a secret".to_string());
    }

    let sections = job_sections(workflow);
    let required = ["check", "check-hosted", "windows-acl"];
    if required.iter().any(|name| !sections.contains_key(*name)) {
        failures.push("required ABI CI jobs are missing".to_string());
        return dedup(failures);
    }

    for sibling in siblings.keys() {
        let checkout = format!("repository: {owner}/{sibling}");
        let env_name = format!("{}_REVISION", sibling.to_uppercase().replace('-', "_"));
        let total: usize = sections
            .values()
            .map(|s| s.matches(&checkout).count())
            .sum();
        if total != required.len() {
            failures.push(format!(
                "every ABI CI job must check out the required {sibling} repository once"
            ));
        }
        for name in required {
            let section = &sections[name];
            let count = section.matches(&checkout).count();
            let ref_pat = format!("ref: ${{{{ env.{env_name} }}}}");
            if count != 1 || !section.contains(&ref_pat) {
                failures.push(format!("{name} must use the immutable {sibling} checkout"));
            }
            let path_pat = format!("path: {sibling}");
            if !section.contains(&path_pat) {
                failures.push(format!("{name} must place {sibling} at the sibling path"));
            }
        }
    }

    failures.extend(validate_checkout_credentials(workflow));

    let trusted = &sections["check"];
    if !trusted.contains("runs-on: [self-hosted")
        || !trusted.contains("github.event.pull_request.head.repo.full_name == github.repository")
    {
        failures.push(
            "trusted self-hosted job must require a same-repository pull request".to_string(),
        );
    }

    let hosted = &sections["check-hosted"];
    let hosted_re = Regex::new(r"(?m)^    runs-on:\s*([^\n#]+)").unwrap();
    let hosted_runner = hosted_re
        .captures(hosted)
        .map(|c| c.get(1).unwrap().as_str().trim().to_string());
    let has_fork_check =
        hosted.contains("github.event.pull_request.head.repo.full_name != github.repository");
    let valid_runners: BTreeSet<&str> = ["macos-latest", "ubuntu-latest", "windows-latest"]
        .into_iter()
        .collect();
    let runner_ok = hosted_runner
        .as_ref()
        .is_some_and(|r| valid_runners.contains(r.as_str()));
    if !has_fork_check || hosted_runner.is_none() || !runner_ok {
        failures.push("fork pull requests must run on a GitHub-hosted runner".to_string());
    }

    dedup(failures)
}

fn dedup(input: Vec<String>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for item in input {
        if seen.insert(item.clone()) {
            out.push(item);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_requirements_matches_python_logic() {
        let cargo = r#"
[workspace.dependencies]
abi-wdbx = { path = "../wdbx/crates/abi-wdbx" }
abi-core = { path = "../wdbx/crates/abi-core" }
local = { path = "crates/local" }
other = { version = "1.0" }
"#;
        let req = sibling_dependency_requirements(cargo);
        assert_eq!(req.len(), 1);
        assert!(req.contains_key("wdbx"));
        assert_eq!(
            req["wdbx"],
            vec!["../wdbx/crates/abi-core", "../wdbx/crates/abi-wdbx"]
        );
    }

    #[test]
    fn repository_owner_parses_github_url() {
        let cargo = r#"
[workspace.package]
repository = "https://github.com/donaldfilimon/abi"
[workspace.dependencies]
"#;
        assert_eq!(repository_owner(cargo).unwrap(), "donaldfilimon");
    }

    #[test]
    fn yaml_indirection_detection_matches_python() {
        assert!(has_yaml_indirection("uses: *checkout"));
        assert!(has_yaml_indirection("uses: &anchor value"));
        assert!(!has_yaml_indirection("uses: actions/checkout@abc"));
        assert!(has_yaml_indirection("  - uses: *1"));
        // word char preceding should not trigger
        assert!(!has_yaml_indirection("foo&bar"));
    }
}
