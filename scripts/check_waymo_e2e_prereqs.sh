#!/usr/bin/env bash
# Pre-flight for the Waymo preprocess path.
# Mirrors scripts/check_mcd_e2e_prereqs.sh in intent: print what is present vs
# missing so the user can decide whether `3dgs-robotics run --preprocess-method
# waymo` is runnable on this machine.

set -u
status=0

section() {
    printf "\n=== %s ===\n" "$1"
}

check() {
    local label=$1
    local cmd=$2
    if eval "$cmd" >/dev/null 2>&1; then
        printf "  [OK]   %s\n" "$label"
    else
        printf "  [MISS] %s\n" "$label"
        status=1
    fi
}

section "Python runtime"
python_ver=$(python3 -c 'import sys; print(".".join(str(x) for x in sys.version_info[:2]))' 2>/dev/null || echo "?")
printf "  python3 = %s\n" "${python_ver}"
if [ "${python_ver}" = "3.10" ] || [ "${python_ver}" = "3.11" ]; then
    printf "  [OK]   Python is compatible with waymo-open-dataset-tf-2-12-0\n"
else
    printf "  [WARN] waymo-open-dataset-tf-2-12-0 targets Python 3.10. "
    printf "Use pyenv or a matching venv.\n"
    status=1
fi

section "Editable repo install"
check "gs_sim2real importable" "python3 -c 'import gs_sim2real'"
check "3dgs-robotics CLI on PATH"  "command -v 3dgs-robotics"

section "Waymo SDK"
check "waymo_open_dataset importable" \
    "python3 -c 'import waymo_open_dataset'"
check "tensorflow importable" \
    "python3 -c 'import tensorflow'"

section "Input data"
data_dir=${WAYMO_DATA_DIR:-data/waymo}
printf "  WAYMO_DATA_DIR = %s\n" "${data_dir}"
if compgen -G "${data_dir}/*.tfrecord" >/dev/null; then
    count=$(find "${data_dir}" -maxdepth 1 -name "*.tfrecord" | wc -l)
    printf "  [OK]   %s tfrecord files in %s\n" "${count}" "${data_dir}"
else
    printf "  [MISS] no *.tfrecord in %s\n" "${data_dir}"
    printf "         Download from https://waymo.com/open/download/ after "
    printf "agreeing to the dataset terms of use.\n"
    status=1
fi

section "Summary"
if [ ${status} -eq 0 ]; then
    printf "  All prerequisites present. Run: 3dgs-robotics run --preprocess-method waymo --images %s ...\n" "${data_dir}"
else
    printf "  One or more prerequisites missing. See above [MISS]/[WARN] lines.\n"
fi
exit ${status}
