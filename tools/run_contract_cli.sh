#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=contract_cli/common.sh
source "$ROOT/contract_cli/common.sh"
# shellcheck source=contract_cli/help.sh
source "$ROOT/contract_cli/help.sh"
# shellcheck source=contract_cli/complete_through_wdbx.sh
source "$ROOT/contract_cli/complete_through_wdbx.sh"
# shellcheck source=contract_cli/dashboard_tui.sh
source "$ROOT/contract_cli/dashboard_tui.sh"
# shellcheck source=contract_cli/nn.sh
source "$ROOT/contract_cli/nn.sh"
# shellcheck source=contract_cli/agent_orchestration.sh
source "$ROOT/contract_cli/agent_orchestration.sh"

echo "run_contract_cli: ok"
