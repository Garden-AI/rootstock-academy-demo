# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mp-api>=0.41",
#     "pymatgen>=2024.1",
#     "ase>=3.22",
# ]
# ///
"""Fetch committee-demo seed structures from the Materials Project.

One-time step, run anywhere with internet access (laptop or a login node —
compute nodes can't reach MP). Writes an extxyz of the stable and
near-stable phases of a chemical system, with MP provenance in each frame's
info fields; the committee demo (academy_mlip_committee.py) reads that file
and never talks to MP itself.

    export MP_API_KEY=...   # free key from materialsproject.org/api
    uv run fetch_mp_structures.py                 # Cu-Au, writes mp_seeds.extxyz
    uv run fetch_mp_structures.py --chemsys Cu-Ni --out cuni_seeds.extxyz

Default system Cu-Au: the classic order-disorder alloy (L1_2 Cu3Au / L1_0
CuAu / L1_2 CuAu3 plus the fcc elements), well represented in MP and firmly
in-domain for MP-trained foundation MLIPs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chemsys", default="Cu-Au",
                        help="binary system, e.g. Cu-Au (default: Cu-Au)")
    parser.add_argument("--ehull-max", type=float, default=0.05,
                        help="max energy above hull, eV/atom (default: 0.05)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output extxyz (default: mp_seeds.extxyz next to "
                             "this script)")
    parser.add_argument("--api-key", default=None,
                        help="MP API key (default: MP_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("need a Materials Project API key: set MP_API_KEY or pass "
                 "--api-key (free key: https://materialsproject.org/api)")
    out = args.out or Path(__file__).parent / "mp_seeds.extxyz"

    from ase.io import write as ase_write
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    elements = args.chemsys.split("-")
    chemsys_queries = elements + [args.chemsys]  # endpoints + the binary

    print(f"querying MP for {chemsys_queries}, e_above_hull <= {args.ehull_max} eV/atom")
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=chemsys_queries,
            energy_above_hull=(0.0, args.ehull_max),
            fields=["material_id", "formula_pretty", "structure",
                    "energy_above_hull", "symmetry", "theoretical"],
        )

    # One seed per composition: the lowest-hull entry (the ordered ground
    # state; polymorphs above it add little to a disagreement-hunting demo).
    best = {}
    for doc in docs:
        key = doc.formula_pretty
        if key not in best or doc.energy_above_hull < best[key].energy_above_hull:
            best[key] = doc

    frames = []
    for doc in sorted(best.values(), key=lambda d: d.formula_pretty):
        structure = SpacegroupAnalyzer(doc.structure).get_conventional_standard_structure()
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.info.update({
            "seed_name": doc.formula_pretty,
            "mp_id": str(doc.material_id),
            "e_above_hull": round(float(doc.energy_above_hull), 4),
            "spacegroup": doc.symmetry.symbol if doc.symmetry else "?",
            "source": f"Materials Project {doc.material_id}",
        })
        frames.append(atoms)
        print(f"  {doc.formula_pretty:<8s} {doc.material_id:<12s} "
              f"{atoms.info['spacegroup']:<10s} "
              f"e_hull={doc.energy_above_hull:.4f} eV/atom  "
              f"{len(atoms)} atoms (conventional cell)")

    if len(frames) < 3:
        sys.exit(f"only {len(frames)} phases found for {args.chemsys} — "
                 "widen --ehull-max or pick a richer system")
    ase_write(out, frames)
    print(f"wrote {len(frames)} seeds to {out}")


if __name__ == "__main__":
    main()
