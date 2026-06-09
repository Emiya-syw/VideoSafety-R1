#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Each stage consumes the final checkpoint directory produced by the preceding
# stage. Run an individual stage directly when resuming a partially completed
# recipe.
"${SCRIPT_DIR}/stage1_alarm_ar.sh"
"${SCRIPT_DIR}/stage2_alarm_atc.sh"
"${SCRIPT_DIR}/stage3_cot.sh"
"${SCRIPT_DIR}/stage4_grpo.sh"
