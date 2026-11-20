#!/usr/bin/env python3
"""Generate the inputs of the fixedCutOff test (cases/<name>/simulation.json).

Isolated pairs of particles at known distances (types A and B, so sigma_ij differs between
pairs), one step with the "None" integrator, PotentialMeasure of every particle.
check.py recomputes energies and forces from each simulation.json and compares.

Only the Python standard library is used. Run from this folder: python3 generate.py
"""
import json
import math
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))

BOX = 200.0       # pairs sit on a grid with spacing SPACING, far apart (no interaction between pairs)
SPACING = 25.0

# SplitLennardJones / LennardJonesType1 pair parameters: name_i, name_j, epsilon, sigma
LJ_DATA = [["A", "A", 1.0, 1.0],
           ["A", "B", 0.8, 1.25],
           ["B", "B", 0.6, 1.5]]
LJ_FACTOR = 2.5   # scaled cutoffs 2.5, 3.125, 3.75
LJ_FIXED = 3.0    # between them: AA pairs gain range, BB pairs lose it

DH_LENGTH = 1.0
DH_FACTOR = 3.0   # cutoff 3.0
DH_FIXED = 4.0


def distances(sigma, rmax, boundaries):
    """Distances from 0.95 sigma to rmax, 0.1 apart, avoiding +-0.02 around any boundary
    (cutoffs, 2^(1/6) sigma), where single precision could fall on either side."""
    out = []
    r = 0.95 * sigma
    while r <= rmax:
        if all(abs(r - b) > 0.02 for b in boundaries):
            out.append(round(r, 6))
        r += 0.1
    return out


def unit_vector(rng):
    while True:
        v = [rng.uniform(-1, 1) for _ in range(3)]
        n = math.sqrt(sum(x * x for x in v))
        if 0.1 < n <= 1.0:
            return [x / n for x in v]


def build_pairs(pair_list):
    """pair_list: (type_i, type_j, r). Returns state data and structure data."""
    rng = random.Random(12345)
    n_side = int(BOX // SPACING)
    slots = [(i, j, k) for i in range(n_side) for j in range(n_side) for k in range(n_side)]
    assert len(pair_list) <= len(slots), "too many pairs for the box"
    state, structure = [], []
    pid = 0
    for (ti, tj, r), (a, b, c) in zip(pair_list, slots):
        center = [-BOX / 2 + SPACING * (a + 0.5), -BOX / 2 + SPACING * (b + 0.5), -BOX / 2 + SPACING * (c + 0.5)]
        u = unit_vector(rng)
        p1 = [center[d] - 0.5 * r * u[d] for d in range(3)]
        p2 = [center[d] + 0.5 * r * u[d] for d in range(3)]
        for t, p in ((ti, p1), (tj, p2)):
            state.append([pid, [round(x, 6) for x in p]])
            structure.append([pid, t])
            pid += 1
    return state, structure


def lj_pairs():
    pairs = []
    for ti, tj, eps, sig in LJ_DATA:
        bounds = [LJ_FACTOR * sig, LJ_FIXED, 2 ** (1 / 6) * sig]
        pairs += [(ti, tj, r) for r in distances(sig, 4.5, bounds)]
    return pairs


def dh_pairs():
    pairs = []
    for ti, tj in (("A", "A"), ("A", "B"), ("B", "B")):
        pairs += [(ti, tj, r) for r in distances(0.6, 5.0, [DH_FACTOR * DH_LENGTH, DH_FIXED])]
    return pairs


def simulation(name, pairs, potential):
    state, structure = build_pairs(pairs)
    ids = [[s[0]] for s in state]
    return {
        "system": {"info": {"type": ["Simulation", "Information"], "parameters": {"name": name}}},
        "global": {
            "units": {"type": ["Units", "None"]},
            "types": {"type": ["Types", "Basic"],
                      "labels": ["name", "mass", "radius", "charge"],
                      "data": [["A", 1.0, 0.5, 1.0], ["B", 1.0, 0.5, -1.0]]},
            "ensemble": {"type": ["Ensemble", "NVT"], "labels": ["box", "temperature"],
                         "data": [[[BOX, BOX, BOX], 1.0]]},
        },
        "integrator": {
            "none": {"type": ["None", "None"]},
            "schedule": {"type": ["Schedule", "Integrator"], "labels": ["order", "integrator", "steps"],
                         "data": [[1, "none", 1]]},
        },
        "state": {"labels": ["id", "position"], "data": state},
        "topology": {
            "structure": {"labels": ["id", "type"], "data": structure},
            "forceField": {
                "nl": {"type": ["VerletConditionalListSet", "all"], "parameters": {"cutOffVerletFactor": 1.1}},
                "pot": potential,
            },
        },
        "simulationStep": {
            "potential": {"type": ["ParticlesListMeasure", "PotentialMeasure"],
                          "parameters": {"intervalStep": 1, "outputFilePath": "pot.dat"},
                          "labels": ["id"], "data": ids},
        },
    }


def split_lj(cut):
    par = {"condition": "all", "epsilon_r": 1.0, "epsilon_a": 0.7}
    par.update(cut)
    return {"type": ["NonBonded", "SplitLennardJones"], "parameters": par,
            "labels": ["name_i", "name_j", "epsilon", "sigma"], "data": LJ_DATA}


def lj1(cut):
    par = {"condition": "all"}
    par.update(cut)
    return {"type": ["NonBonded", "LennardJonesType1"], "parameters": par,
            "labels": ["name_i", "name_j", "epsilon", "sigma"], "data": LJ_DATA}


def dh(cut):
    par = {"condition": "all", "debyeLength": DH_LENGTH, "dielectricConstant": 2.0}
    par.update(cut)
    return {"type": ["NonBonded", "DebyeHuckel"], "parameters": par}


# name: (pairs, potential, expected error text or None)
CASES = {
    "splitLJ_factor": (lj_pairs, split_lj({"cutOffFactor": LJ_FACTOR}), None),
    "splitLJ_fixed":  (lj_pairs, split_lj({"cutOff": LJ_FIXED}), None),
    "LJ1_fixed":      (lj_pairs, lj1({"cutOff": LJ_FIXED}), None),
    "DH_factor":      (dh_pairs, dh({"cutOffFactor": DH_FACTOR}), None),
    "DH_fixed":       (dh_pairs, dh({"cutOff": DH_FIXED}), None),
    "error_both":     (lj_pairs, split_lj({"cutOffFactor": LJ_FACTOR, "cutOff": LJ_FIXED}), "Exactly one of the parameters"),
    "error_none":     (lj_pairs, split_lj({}), "Exactly one of the parameters"),
    "error_negative": (lj_pairs, split_lj({"cutOff": -1.0}), "must be positive"),
}

if __name__ == "__main__":
    for name, (pairs, potential, expected_error) in CASES.items():
        d = os.path.join(HERE, "cases", name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "simulation.json"), "w") as f:
            text = json.dumps(simulation(name, pairs(), potential), indent=1)
            # short lists (one particle, one pair parameter set, ...) on a single line
            one_line = lambda m: " ".join(m.group(0).split()) if len(" ".join(m.group(0).split())) < 100 else m.group(0)
            text = re.sub(r"\[[^\[\]{}]*\]", one_line, text)
            text = re.sub(r"\[(?:[^\[\]{}]|\[[^\[\]{}]*\])*\]", one_line, text)
            f.write(text + "\n")
        err_file = os.path.join(d, "expect_error")
        if expected_error:
            with open(err_file, "w") as f:
                f.write(expected_error + "\n")
        elif os.path.exists(err_file):
            os.remove(err_file)
        print("written", os.path.relpath(d, HERE), "(expects error)" if expected_error else "")
