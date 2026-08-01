#!/usr/bin/env bash
#
# run_checks.sh — the gate for "is this stage done?"
#
# Two of the four Definition-of-Done conditions in CLAUDE.md are mechanical:
# tests must be green, and every notebook must run top-to-bottom from a clean
# kernel. This script checks both. The other two (interview notes written,
# README results table updated) are human checks and are printed as a reminder
# at the end rather than faked here.
#
# Notebooks are executed with --inplace so that the committed .ipynb contains
# the outputs a reader sees on GitHub. A notebook that only runs because of
# stale state left in the kernel is a broken notebook, so nbconvert always
# starts fresh.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# Which interpreter runs everything. Override with PYTHON=... if your
# environment differs. The default is not arbitrary: on this machine a Windows
# Application Control policy blocks torch's DLLs in the base Python 3.13
# install (the block is file-based, so a venv does not work around it), and the
# py310 conda environment is the one where torch actually loads.
PYTHON="${PYTHON:-D:/miniconda/envs/py310/python.exe}"
if [ ! -x "$PYTHON" ] && ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "note: $PYTHON not found, falling back to 'python' on PATH"
    PYTHON=python
fi
echo "interpreter: $("$PYTHON" -c 'import sys; print(sys.version.split()[0], sys.executable)')"

# jupyter_core locks down the kernel connection file on Windows via
# win32security, which some managed machines block with an Application Control
# policy — the kernel then fails to launch at all. The connection file lives in
# a per-user runtime dir and holds no secrets we care about here, so allowing
# the plain write is the right trade for a local test runner.
export JUPYTER_ALLOW_INSECURE_WRITES=true

echo "=== 1/3  pytest ==============================================="
"$PYTHON" -m pytest tests/ -v

echo
echo "=== 2/3  C++ parity and equivalence ==========================="

# From Stage 7 the C++ inference path carries two MANDATORY tests: parity
# against PyTorch, and the streaming path against the full recompute. They live
# in ctest rather than pytest because they are C++, but they are part of the
# same gate.
#
# Skipped rather than failed when the toolchain is absent, because a machine
# with no compiler can still legitimately run the Python half — but the skip is
# printed loudly, since a silently skipped mandatory test is worse than no test.
if ! command -v cmake >/dev/null 2>&1; then
    echo "SKIPPED: cmake not on PATH — the C++ parity and equivalence tests did NOT run"
elif [ ! -f inference_cpp/artifacts/student_weights.ttsw ]; then
    echo "SKIPPED: no artefacts — run '$PYTHON -m ml.export_weights --real-data' first"
else
    # debug-O0 is the reference and release-avx2 is what the benchmarks use;
    # running both is what proves the optimiser and the intrinsics did not
    # change the answer. The two middle presets are exercised in CI, not here,
    # to keep the gate under a minute.
    for preset in debug-O0 release-avx2; do
        echo "--- $preset"
        (cd inference_cpp \
            && cmake --preset "$preset" >/dev/null \
            && cmake --build --preset "$preset" >/dev/null \
            && ctest --preset "$preset" --output-on-failure | tail -3)
    done
fi

echo
echo "=== 3/3  notebook execution ==================================="

# Collect notebooks explicitly so we can (a) skip checkpoint copies and
# (b) report honestly when there are none, instead of silently passing.
notebooks=()
while IFS= read -r nb; do
    notebooks+=("$nb")
done < <(find notebooks -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' | sort)

if [ ${#notebooks[@]} -eq 0 ]; then
    echo "No notebooks found under notebooks/ — nothing to execute."
else
    for nb in "${notebooks[@]}"; do
        echo "--- executing: $nb"
        # `python -m nbconvert`, not `python -m jupyter nbconvert`: the latter
        # needs the jupyter-nbconvert console script on PATH, which pip does
        # not always put there on Windows.
        "$PYTHON" -m nbconvert \
            --to notebook \
            --execute \
            --inplace \
            --ExecutePreprocessor.timeout=1800 \
            "$nb"
    done
fi

echo
echo "=== mechanical checks passed =================================="
echo "Still required by the Definition of Done, and not checkable here:"
echo "  [ ] docs/INTERVIEW_NOTES_stageN.md written for this stage"
echo "  [ ] README.md 'Headline numbers' table updated from benchmarks/"
