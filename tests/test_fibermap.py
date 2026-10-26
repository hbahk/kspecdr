import os
import sys
import logging
import tempfile

import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src'))

from kspecdr.inst.fibermap import (
    hole_label,
    hole_number,
    make_fiber_table,
    read_fibermap,
    slit_fibers,
)

logging.basicConfig(level=logging.INFO)


def check(cond, label):
    print(f"{'SUCCESS' if cond else 'FAIL'} [{label}]")
    return bool(cond)


def test_fibermap():
    fm = read_fibermap()
    ok = check(len(fm) == 76, "73 positioner holes + 3 fixed sky holes")
    ok &= check(sorted(fm["HOLE"][fm["HOLE"] > 0]) == list(range(1, 74)), "holes 1..73")
    ok &= check(int(fm["LOCKED"].sum()) == 28, "28 locked positioners")
    ok &= check(hole_label(1, fm) == "C4" and hole_label(73, fm) == "M5", "hole -> label")
    ok &= check(hole_number("F9", fm) == 23 and hole_number("b7", fm) == 0, "label -> hole")
    ok &= check(all(hole_number(hole_label(h, fm), fm) == h for h in range(1, 74)), "round trip")

    slit = slit_fibers(fm)
    ok &= check(list(slit["SLIT"]) == list(range(1, 44)), "slit 1..43")
    ok &= check(np.sum(slit["ROLE"] == "obj") == 37 and np.sum(slit["ROLE"] == "sky") == 6,
                "37 obj + 6 sky")
    ok &= check(list(slit["LABEL"][[0, 32, 42]]) == ["C4", "K2", "N7"], "slit order")
    return ok


def test_fiber_table():
    fm = read_fibermap()
    # Synthetic assign file: a target on every positioner hole except D3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "TEST_2627_0001.assign.txt")
        with open(path, "w") as f:
            for row in fm[fm["HOLE"] > 0]:
                if row["LABEL"] == "D3":
                    continue
                cls = 8 if row["LABEL"] == "C5" else 1
                f.write(f"{row['LABEL']} 1.0 2.0 13.0 {cls} {row['HOLE']} "
                        f"{200 + row['HOLE']:.3f} -30.0\n")
        tab = make_fiber_table(path, fm, class_types={8: "C"})

    ok = check(len(tab) == 43, "43 fibres")
    types = dict(zip(tab["NAME"], tab["TYPE"]))
    ok &= check(types["C4"] == "P" and types["C5"] == "C" and types["D3"] == "N",
                "obj types from class / missing target")
    ok &= check(all(types[s] == "S" for s in ("G7", "H8", "J6", "B7", "H1", "N7")),
                "sky fibres from the map")
    ok &= check(np.isclose(tab["RA"][0], 201.0) and np.isnan(tab["RA"][14]),
                "coordinates follow the hole label")

    cal = make_fiber_table(None, fm)
    ok &= check(np.sum(cal["TYPE"] == "S") == 6 and np.sum(cal["TYPE"] == "P") == 37,
                "no assign file")
    return ok


if __name__ == "__main__":
    results = [test_fibermap(), test_fiber_table()]
    print("ALL PASSED" if all(results) else "SOME TESTS FAILED")
    sys.exit(0 if all(results) else 1)
