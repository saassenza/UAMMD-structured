# Development tests

Quick GPU tests for changes made on the `develop` branch, run on the cluster before promoting
`develop` to `main`. They are not part of upstream's `Tests/` suite.

Each test is a folder `<test>/` with:
- `cases/<case>/simulation.json`: inputs (committed, so running needs only bash and the program);
- `cases/<case>/expect_error` (optional): the case must FAIL, with this text in its log;
- `generate.py` (optional): regenerates `cases/` (`python3 generate.py`);
- `check.py` (optional): compares the outputs with reference values (exit code 0 = pass).

## Running (Rocinante)

```bash
cp -r ~/sw/uammd-structured-dev/src/devtests ~/uammd_devtests_$(date +%F)    # work on a copy
cd ~/uammd_devtests_<date>
sbatch -w node-28 run_devtests.sh                 # all tests, development install
sbatch -w node-28 run_devtests.sh fixedCutOff     # selected tests
UAMMD=$HOME/sw/uammd-structured/bin/UAMMDlauncher sbatch -w node-28 run_devtests.sh   # another program
```
The result is in `summary_<jobid>.txt` (also in `stdout_<jobid>.log`): one PASS/FAIL line per case,
then the output of each `check.py`. If the node has no `python3`, copy the folder back and run
`python3 check.py` inside the test folder.

## Tests

- `fixedCutOff`: the `cutOff` alternative to `cutOffFactor` in non-bonded potentials. Isolated pairs
  (types A, B with different sigma) at known distances; energies and forces of SplitLennardJones,
  LennardJonesType1 and DebyeHuckel compared with the formulas, with scaled and fixed cutoffs (the
  fixed cases are also checked to differ from the scaled result); invalid inputs (both parameters,
  neither, negative cutoff) must stop with a clear error.
