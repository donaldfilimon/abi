# Contract CLI helpers — sourced by tools/run_contract_cli.sh
export ABI_WDBX_PERSIST=0

if [[ -n "${ABI_EXE:-}" ]]; then
  ABI="$ABI_EXE"
elif [[ -x target/release/abi ]]; then
  ABI=target/release/abi
elif [[ -x target/debug/abi ]]; then
  ABI=target/debug/abi
else
  ABI=target/debug/abi
fi

if [[ ! -x "$ABI" ]]; then
  echo "error: $ABI not found; run './tools/cargo.sh build -p abi-cli' first" >&2
  exit 1
fi

require_substring() {
  local output="$1"
  local needle="$2"
  if ! grep -Fq -- "$needle" <<<"$output"; then
    echo "expected substring missing: $needle" >&2
    echo "output:" >&2
    echo "$output" >&2
    exit 1
  fi
}

reject_substring() {
  local output="$1"
  local needle="$2"
  if grep -Fq -- "$needle" <<<"$output"; then
    echo "unexpected substring present: $needle" >&2
    echo "output:" >&2
    echo "$output" >&2
    exit 1
  fi
}

expect_exit_2() {
  local label="$1"
  local rc="$2"
  local output="$3"
  if [[ "$rc" -ne 2 ]]; then
    echo "expected '$label' to exit 2, got $rc" >&2
    echo "$output" >&2
    exit 1
  fi
}
