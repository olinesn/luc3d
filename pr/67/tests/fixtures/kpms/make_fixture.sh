#!/usr/bin/env bash
# What a real keypoint-MoSeq fit writes, for tests/test-kpms-io.mjs §6 and
# tests/e2e/behavior-kpms-roundtrip.mjs. Regenerates, in this folder:
#   luc3d-kpms.yml   the manifest the fit read (LUC3D's, from make_poses.mjs)
#   results.h5       keypoint-MoSeq's extract_results file
#   results/         its save_results_as_csv folder
# make_poses.mjs writes synthetic poses as LUC3D exports them (two animals ×
# 300 frames at 30 fps, anipose CSVs + luc3d-kpms.yml); this runs the `calls`
# block of that luc3d-kpms.yml as written, except 5 + 5 iterations instead
# of 50 + 500. Only keypoint-MoSeq's OUTPUT is kept here, never its code (its
# licence is academic-only). Made with keypoint-moseq 0.8.4 (jax 0.6.2, CPU,
# macOS).
#
#   bash tests/fixtures/kpms/make_fixture.sh <python of an environment with keypoint-moseq>
set -euo pipefail
PY="${1:?usage: make_fixture.sh <python with keypoint-moseq>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
node "$HERE/make_poses.mjs" "$WORK"
"$PY" - "$WORK" <<'EOF'
import os, sys, yaml
work = sys.argv[1]
calls = yaml.safe_load(open(os.path.join(work, 'luc3d-kpms.yml')))['calls']
assert 'AR_ITERS, FULL_ITERS = 50, 500' in calls
calls = calls.replace('AR_ITERS, FULL_ITERS = 50, 500', 'AR_ITERS, FULL_ITERS = 5, 5')
os.chdir(work)
exec(compile(calls, 'luc3d-kpms.yml (calls)', 'exec'), {})
EOF
RESULTS="$(find "$WORK/kpms_project" -name results.h5 | head -1)"
cp "$WORK/luc3d-kpms.yml" "$RESULTS" "$HERE/"
rm -rf "$HERE/results"
cp -R "$(dirname "$RESULTS")/results" "$HERE/results"
echo "keypoint-moseq $("$PY" -c 'import keypoint_moseq; print(keypoint_moseq.__version__)')"
