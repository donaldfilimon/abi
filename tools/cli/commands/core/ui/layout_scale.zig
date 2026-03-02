//! Shared scaling helpers for command dashboards and compact TUI views.

pub fn safeSub(value: usize, amount: usize) usize {
    return value -| amount;
}

pub fn innerWidth(cols: u16) usize {
    return safeSub(@as(usize, cols), 2);
}

pub fn clampDimension(value: u16, ideal: usize, margin: usize) usize {
    const budget = safeSub(@as(usize, value), margin);
    return @min(ideal, budget);
}

pub fn halfWidth(cols: u16) usize {
    return @as(usize, cols) / 2;
}
