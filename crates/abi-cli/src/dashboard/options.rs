//! Argument parsing and pane/interval validation for `abi dashboard`.

use super::{Format, MAX_REFRESH_MS, MIN_REFRESH_MS, Options, PANES};

pub(super) fn pane_index_for_token(token: &str) -> Option<usize> {
    if token.len() == 1 {
        let key = token.as_bytes()[0];
        for (idx, (_, _, hotkey)) in PANES.iter().enumerate() {
            if *hotkey as u8 == key {
                return Some(idx);
            }
        }
    }
    for (idx, (name, _, _)) in PANES.iter().enumerate() {
        if token.eq_ignore_ascii_case(name) {
            return Some(idx);
        }
        if *name == "storage" && token.eq_ignore_ascii_case("wdbx") {
            return Some(idx);
        }
    }
    None
}

pub(super) fn valid_refresh_interval(raw: u64) -> Option<i32> {
    let ms = i32::try_from(raw).ok()?;
    if (MIN_REFRESH_MS..=MAX_REFRESH_MS).contains(&ms) {
        Some(ms)
    } else {
        None
    }
}

pub(super) fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut i = 0;
    while i < args.len() {
        let tok = args[i].as_str();
        match tok {
            "--plain" | "--no-color" => options.color = false,
            "--compact" => options.compact = true,
            "--once" => options.force_one_shot = true,
            "--json" => options.format = Format::Json,
            "--list-panes" => options.list_panes = true,
            "--pane" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    return Err("missing --pane value".into());
                };
                let Some(idx) = pane_index_for_token(value) else {
                    return Err(format!("unknown pane '{value}'"));
                };
                options.initial_pane = idx;
            }
            "--interval" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    return Err("missing --interval value".into());
                };
                let Ok(raw) = value.parse::<u64>() else {
                    return Err(format!("invalid --interval '{value}'"));
                };
                let Some(ms) = valid_refresh_interval(raw) else {
                    return Err(format!(
                        "--interval must be {MIN_REFRESH_MS}-{MAX_REFRESH_MS} ms"
                    ));
                };
                options.refresh_interval_ms = ms;
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown flag '{flag}'"));
            }
            other => {
                return Err(format!("unexpected argument '{other}'"));
            }
        }
        i += 1;
    }
    Ok(options)
}
