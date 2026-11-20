#!/bin/bash
# Run the development tests (devtests/<test>/cases/<case>/simulation.json) on one GPU, judge them, and
# write a summary. Use a COPY of the devtests folder (outputs are written next to the inputs):
#     cp -r ~/sw/uammd-structured-dev/src/devtests ~/uammd_devtests_$(date +%F)
#     cd ~/uammd_devtests_<date> && sbatch -w node-28 run_devtests.sh             # all tests
#     sbatch -w node-28 run_devtests.sh fixedCutOff                              # only some tests
# Program: $UAMMD (default: the development install), e.g. UAMMD=$HOME/sw/uammd-structured/bin/UAMMDlauncher sbatch ...
# A case passes if it runs (exit 0), or, if the case has a file "expect_error", if it fails and its
# log contains the text of that file. Then each test's check.py (if any) compares the outputs.
#SBATCH --job-name=uammd_devtests
#SBATCH --partition=spmth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=node-27
#SBATCH --time=00:30:00
#SBATCH --output=stdout_%j.log
#SBATCH --error=stderr_%j.log

if [[ $0 == */slurm_script ]]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")"; fi
USESCRATCH_EXCLUDE=('/stdout_*.log' '/stderr_*.log' '/summary_*.txt')
[ -f /usr/local/bin/usescratch ] && source /usr/local/bin/usescratch   # mandatory on Rocinante

UAMMD=${UAMMD:-$HOME/sw/uammd-structured-dev/bin/UAMMDlauncher}
SUMMARY=summary_${SLURM_JOB_ID:-local}.txt
TESTS=("$@")
[ ${#TESTS[@]} -eq 0 ] && TESTS=($(for d in */cases; do dirname "$d"; done))

{
  echo "date: $(date '+%F %T')   node: $(hostname -s)   GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
  echo "program: $UAMMD"
  grep -E '^(source|GPU archs)' "$(dirname "$UAMMD")/../INSTALL_INFO.txt" 2>/dev/null
  echo
} > "$SUMMARY"

n_fail=0
for t in "${TESTS[@]}"; do
  echo "== test $t" >> "$SUMMARY"
  for c in "$t"/cases/*/; do
    c=${c%/}; name=$(basename "$c")
    rm -f "$c"/pot.dat "$c"/*.dat "$c"/run.log "$c"/exit_code        # outputs of a previous run (measures append)
    ( cd "$c" && "$UAMMD" simulation.json > run.log 2>&1; echo $? > exit_code )
    code=$(cat "$c/exit_code")
    if [ -f "$c/expect_error" ]; then
      expected=$(head -1 "$c/expect_error")
      if [ "$code" != 0 ] && grep -qF "$expected" "$c/run.log"; then res="PASS (failed as expected: \"$expected\")"
      else res="FAIL (expected an error containing \"$expected\", exit code $code)"; fi
    else
      if [ "$code" = 0 ]; then res="PASS (ran)"; else res="FAIL (exit code $code, see $c/run.log)"; fi
    fi
    [[ $res == FAIL* ]] && n_fail=$((n_fail+1))
    printf '  %-18s %s\n' "$name" "$res" >> "$SUMMARY"
  done
  if [ -f "$t/check.py" ]; then
    if command -v python3 > /dev/null; then
      ( cd "$t" && python3 check.py ) >> "$SUMMARY" 2>&1 || n_fail=$((n_fail+1))
    else
      echo "  (no python3 on this node: copy the folder back and run python3 check.py in $t)" >> "$SUMMARY"
    fi
  fi
  echo >> "$SUMMARY"
done
[ $n_fail = 0 ] && echo "== ALL DEVTESTS PASSED" >> "$SUMMARY" || echo "== $n_fail FAILURE(S)" >> "$SUMMARY"
cat "$SUMMARY"
exit $((n_fail > 0))
