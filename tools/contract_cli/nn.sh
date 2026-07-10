# `abi nn` is the miniature character-level demo trainer. Assert the honest
# help framing and a real inline training run (stable label substrings only,
# never float loss values).
nn_help_out="$("$ABI" help nn 2>&1)"
require_substring "$nn_help_out" "character-level demo trainer"
require_substring "$nn_help_out" "not a production/LLM/distributed trainer"

nn_local_help_out="$("$ABI" nn --help 2>&1)"
require_substring "$nn_local_help_out" "train --jsonl <path>"
require_substring "$nn_local_help_out" "sample --text \"<corpus>\""

nn_train_help_out="$("$ABI" nn train --help 2>&1)"
require_substring "$nn_train_help_out" "usage: abi nn train"
if grep -Fq -- "nn train:" <<<"$nn_train_help_out"; then
  echo "expected 'abi nn train --help' to render help, not train" >&2
  echo "$nn_train_help_out" >&2
  exit 1
fi

nn_sample_help_out="$("$ABI" nn sample --help 2>&1)"
require_substring "$nn_sample_help_out" "usage: abi nn sample"

set +e
nn_bad_out="$("$ABI" nn bogus 2>&1)"
nn_bad_rc=$?
set -e
if [[ "$nn_bad_rc" -ne 2 ]]; then
  echo "expected 'abi nn bogus' to exit 2, got $nn_bad_rc" >&2
  echo "$nn_bad_out" >&2
  exit 1
fi
require_substring "$nn_bad_out" "abi nn train"

set +e
nn_sample_missing_out="$("$ABI" nn sample --text hello 2>&1)"
nn_sample_missing_rc=$?
set -e
if [[ "$nn_sample_missing_rc" -ne 2 ]]; then
  echo "expected 'abi nn sample --text hello' to exit 2, got $nn_sample_missing_rc" >&2
  echo "$nn_sample_missing_out" >&2
  exit 1
fi
require_substring "$nn_sample_missing_out" "abi nn sample --text"

wdbx_cluster_help_out="$("$ABI" wdbx cluster --help 2>&1)"
require_substring "$wdbx_cluster_help_out" "usage: abi wdbx cluster"
wdbx_cluster_help_alias_out="$("$ABI" help wdbx cluster 2>&1)"
require_substring "$wdbx_cluster_help_alias_out" "usage: abi wdbx cluster"

nn_train_out="$("$ABI" nn train "hello hello hello " 2>&1)"
require_substring "$nn_train_out" "nn train: initial_loss="
require_substring "$nn_train_out" "improved="
