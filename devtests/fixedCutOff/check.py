#!/usr/bin/env python3
"""Check the fixedCutOff test: recompute, from each cases/<name>/simulation.json, the energy and
force of every particle and compare with the pot.dat written by UAMMD-structured.

Per particle, UAMMD reports half the pair energy and the full pair force.
Fixed-cutoff cases are also compared with what the scaled cutoff would give, to show that the
option changes the result (pairs between the two cutoffs).
Only the Python standard library is used (works with python >= 3.6).
Run from this folder after the runs: python3 check.py   (exit code 0 = all passed)
"""
import json
import math
import os
import sys

import generate  # same folder: the scaled factors used for the comparison

HERE = os.path.dirname(os.path.abspath(__file__))
ABS_TOL = 2e-5    # pot.dat has 6 decimals; UAMMD runs in single precision
REL_TOL = 2e-4


def read_pot(path):
    """pot.dat -> {id: (energy, [fx, fy, fz])} for the first interactor."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#") or not line.strip():   # header is right-aligned
                continue
            v = line.split()
            out[int(float(v[0]))] = (float(v[1]), [float(v[2]), float(v[3]), float(v[4])])
    return out


def pair_function(pot, types_charge, rc_override=None):
    """Returns f(type_i, type_j, rij_vector, i, j) -> (pair energy, force on i) or None if cut."""
    kind = pot["type"][1]
    par = pot["parameters"]

    if kind in ("SplitLennardJones", "LennardJonesType1"):
        table = {}
        for ti, tj, eps, sig in pot["data"]:
            table[(ti, tj)] = table[(tj, ti)] = (eps, sig)

        def f(ti, tj, rij):
            eps, sig = table[(ti, tj)]
            if rc_override is not None:
                rc = rc_override(sig)
            else:
                rc = par["cutOff"] if "cutOff" in par else par["cutOffFactor"] * sig
            r2 = sum(x * x for x in rij)
            if r2 > rc * rc:
                return None
            if kind == "SplitLennardJones":
                eps_a = eps * par["epsilon_a"]
                e_lj = par["epsilon_r"] if r2 < sig * sig * 1.259921 else eps_a
                shift = e_lj - eps_a
            else:
                e_lj, shift = eps, 0.0
            s6 = (sig * sig / r2) ** 3
            energy = 4 * e_lj * (s6 * s6 - s6) + shift
            fmod = -24 * e_lj * (2 * s6 * s6 - s6) / r2
            return energy, [fmod * x for x in rij]
        return f

    if kind == "DebyeHuckel":
        lam, epsr = par["debyeLength"], par["dielectricConstant"]

        def f(ti, tj, rij):
            if rc_override is not None:
                rc = rc_override(lam)
            else:
                rc = par["cutOff"] if "cutOff" in par else par["cutOffFactor"] * lam
            r2 = sum(x * x for x in rij)
            q = types_charge[ti] * types_charge[tj]
            if r2 > rc * rc or q == 0:
                return None
            r = math.sqrt(r2)
            efac = q / epsr          # Units "None": electric conversion factor 1
            energy = efac * math.exp(-r / lam) / r
            fmod = -efac * math.exp(-r / lam) / r2 * (1 / lam + 1 / r)
            return energy, [fmod * x for x in rij]
        return f

    raise SystemExit("check.py: potential %s not handled" % kind)


def reference(sim, pair):
    pos = {i: p for i, p in sim["state"]["data"]}
    typ = {i: t for i, t in sim["topology"]["structure"]["data"]}
    box = sim["global"]["ensemble"]["data"][0][0]
    ids = sorted(pos)
    ref = {i: [0.0, [0.0, 0.0, 0.0]] for i in ids}
    n_interacting = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            rij = [pos[j][d] - pos[i][d] for d in range(3)]
            rij = [x - box[d] * round(x / box[d]) for d, x in enumerate(rij)]   # minimum image
            res = pair(typ[i], typ[j], rij)
            if res is None:
                continue
            n_interacting += 1
            e, fi = res
            ref[i][0] += e / 2
            ref[j][0] += e / 2
            for d in range(3):
                ref[i][1][d] += fi[d]
                ref[j][1][d] -= fi[d]
    return ref, n_interacting


def max_dev(out, ref):
    de = df = 0.0
    ok = True
    for i, (e_ref, f_ref) in ref.items():
        e, f = out[i]
        de = max(de, abs(e - e_ref))
        ok &= abs(e - e_ref) <= ABS_TOL + REL_TOL * abs(e_ref)
        for d in range(3):
            df = max(df, abs(f[d] - f_ref[d]))
            ok &= abs(f[d] - f_ref[d]) <= ABS_TOL + REL_TOL * abs(f_ref[d])
    return ok, de, df


def main():
    cases_dir = os.path.join(HERE, "cases")
    all_ok = True
    print("%-16s %6s %12s %11s %11s  %s" % ("case", "pairs", "interacting", "max|dE|", "max|dF|", "result"))
    for name in sorted(os.listdir(cases_dir)):
        d = os.path.join(cases_dir, name)
        if os.path.exists(os.path.join(d, "expect_error")):
            continue                      # error cases are judged by the runner (run_devtests.sh)
        with open(os.path.join(d, "simulation.json")) as f:
            sim = json.load(f)
        pot = sim["topology"]["forceField"]["pot"]
        charge = {t[0]: t[3] for t in sim["global"]["types"]["data"]}
        n_pairs = len(sim["state"]["data"]) // 2
        pot_file = os.path.join(d, "pot.dat")
        if not os.path.exists(pot_file):
            print("%-16s %6d %12s %11s %11s  FAIL (no pot.dat: did the run fail? see run.log)" % (name, n_pairs, "-", "-", "-"))
            all_ok = False
            continue
        out = read_pot(pot_file)
        ref, n_int = reference(sim, pair_function(pot, charge))
        ok, de, df = max_dev(out, ref)
        note = ""
        if ok and "cutOff" in pot["parameters"]:
            # the same system with the scaled cutoff must differ (otherwise the test proves nothing)
            factor = generate.DH_FACTOR if pot["type"][1] == "DebyeHuckel" else generate.LJ_FACTOR
            ref_s, n_int_s = reference(sim, pair_function(pot, charge, rc_override=lambda s: factor * s))
            ok_s, _, _ = max_dev(out, ref_s)
            if ok_s:
                ok = False
                note = " (output also matches the SCALED cutoff: the fixed cutoff had no effect)"
            else:
                note = " (scaled cutoff x%g would give %d interacting pairs)" % (factor, n_int_s)
        all_ok &= ok
        print("%-16s %6d %12d %11.2e %11.2e  %s%s" % (name, n_pairs, n_int, de, df, "PASS" if ok else "FAIL", note))
    print("fixedCutOff numerical check:", "ALL PASSED" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
