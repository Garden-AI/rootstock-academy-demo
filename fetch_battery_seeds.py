# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mp-api>=0.41",
#     "pymatgen>=2024.1",
#     "ase>=3.22",
# ]
# ///
"""Fetch a ~25-compound Li-ion battery seed set from the Materials Project.

Cathodes (layered, spinel, olivine, tavorite, silicate, Li-rich), anodes
(Li metal, graphite / LiC6, Li15Si4, Li4Ti5O12) and solid electrolytes
(garnet LLZO, LGPS, thio-LISICON, argyrodite, NASICON, antiperovskite,
Li3N, Li2S). Every entry is a stoichiometric, ordered ground state on or
near the MP hull, so it is in-domain for MP-trained foundation MLIPs — the
committee then probes what those models *don't* agree on: delithiation,
Li hops, and Li/TM antisite disorder (see academy_mlip_committee.py).

One-time step, run anywhere with internet access:

    export MP_API_KEY=...   # free key from materialsproject.org/api
    uv run fetch_battery_seeds.py                     # writes battery_seeds.extxyz
    uv run academy_mlip_committee.py --seeds battery_seeds.extxyz --cluster delta

Formulas resolve to the lowest-hull *experimentally observed* MP entry
(theoretical=False preferred); pin a polymorph with an explicit mp-id.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# (label, MP query, role).  Query is a formula (lowest-hull entry wins) or an
# explicit mp-id to pin a polymorph.  Labels are what the campaign reports.
BATTERY_SET: list[tuple[str, str, str]] = [
    # -- layered / spinel / polyanion cathodes ------------------------------
    ("LiCoO2",       "LiCoO2",        "cathode: layered O3"),
    ("LiNiO2",       "LiNiO2",        "cathode: layered O3"),
    ("LiMnO2",       "LiMnO2",        "cathode: layered/orthorhombic"),
    ("Li2MnO3",      "Li2MnO3",       "cathode: Li-rich layered"),
    ("LiMn2O4",      "LiMn2O4",       "cathode: spinel"),
    ("LiNi0.5Mn1.5O4", "LiNiMn3O8",   "cathode: high-voltage spinel"),
    ("LiFePO4",      "LiFePO4",       "cathode: olivine"),
    ("LiMnPO4",      "LiMnPO4",       "cathode: olivine"),
    ("LiCoPO4",      "LiCoPO4",       "cathode: olivine (5 V)"),
    ("LiVPO4F",      "LiVPO4F",       "cathode: tavorite"),
    ("LiFeSO4F",     "LiFeSO4F",      "cathode: tavorite/triplite"),
    ("Li2FeSiO4",    "Li2FeSiO4",     "cathode: silicate"),
    ("Li3V2(PO4)3",  "Li3V2(PO4)3",   "cathode: NASICON-type"),
    ("LiTiS2",       "LiTiS2",        "cathode: layered sulfide (Whittingham)"),
    ("LiVO2",        "LiVO2",         "cathode: layered"),
    # -- anodes -------------------------------------------------------------
    ("Li",           "Li",            "anode: metal"),
    ("LiC6",         "LiC6",          "anode: lithiated graphite"),
    ("Li15Si4",      "Li15Si4",       "anode: lithiated silicon"),
    ("Li4Ti5O12",    "Li4Ti5O12",     "anode: zero-strain spinel"),
    ("Li7Ti5O12",    "Li7Ti5O12",     "anode: lithiated LTO"),
    # -- solid electrolytes -------------------------------------------------
    ("LLZO",         "Li7La3Zr2O12",  "electrolyte: garnet"),
    ("LGPS",         "Li10Ge(PS6)2",  "electrolyte: LGPS"),
    ("Li3PS4",       "Li3PS4",        "electrolyte: thio-LISICON"),
    ("Li6PS5Cl",     "Li6PS5Cl",      "electrolyte: argyrodite"),
    ("LiTi2(PO4)3",  "LiTi2(PO4)3",   "electrolyte: NASICON"),
    ("Li3OCl",       "Li3ClO",        "electrolyte: antiperovskite"),
    ("Li3N",         "Li3N",          "electrolyte: nitride"),
    ("Li2S",         "Li2S",          "electrolyte / conversion"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="output extxyz (default: battery_seeds.extxyz "
                             "next to this script)")
    parser.add_argument("--ehull-max", type=float, default=0.08,
                        help="max energy above hull, eV/atom (default: 0.08; "
                             "some real cathodes are metastable in MP)")
    parser.add_argument("--max-atoms", type=int, default=120,
                        help="use the primitive cell when the conventional cell "
                             "exceeds this many atoms (default: 120)")
    parser.add_argument("--api-key", default=None,
                        help="MP API key (default: MP_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("need a Materials Project API key: set MP_API_KEY or pass "
                 "--api-key (free key: https://materialsproject.org/api)")
    out = args.out or Path(__file__).parent / "battery_seeds.extxyz"

    from ase.io import write as ase_write
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    fields = ["material_id", "formula_pretty", "structure", "energy_above_hull",
              "symmetry", "theoretical", "is_stable"]
    frames, missing = [], []
    with MPRester(api_key) as mpr:
        for label, query, role in BATTERY_SET:
            if query.startswith("mp-"):
                docs = mpr.materials.summary.search(material_ids=[query], fields=fields)
            else:
                docs = mpr.materials.summary.search(
                    formula=query, energy_above_hull=(0.0, args.ehull_max),
                    fields=fields)
            if not docs:
                missing.append(label)
                print(f"  {label:<14s} — no MP entry within e_hull <= {args.ehull_max}")
                continue
            # Prefer experimentally observed, then lowest hull.
            doc = min(docs, key=lambda d: (bool(d.theoretical), d.energy_above_hull))
            sga = SpacegroupAnalyzer(doc.structure)
            structure = sga.get_conventional_standard_structure()
            if len(structure) > args.max_atoms:
                structure = sga.get_primitive_standard_structure()
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.info.update({
                "seed_name": label,
                "role": role,
                "mp_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "e_above_hull": round(float(doc.energy_above_hull), 4),
                "spacegroup": doc.symmetry.symbol if doc.symmetry else "?",
                "theoretical": bool(doc.theoretical),
                "source": f"Materials Project {doc.material_id}",
            })
            frames.append(atoms)
            print(f"  {label:<14s} {doc.material_id:<12s} {atoms.info['spacegroup']:<10s} "
                  f"e_hull={doc.energy_above_hull:.3f}  {len(atoms):3d} atoms  "
                  f"{'(theoretical)' if doc.theoretical else ''}  {role}")

    if len(frames) < 10:
        sys.exit(f"only {len(frames)} seeds resolved — check API key / network")
    ase_write(out, frames)
    print(f"\nwrote {len(frames)} seeds to {out}"
          + (f"; missing: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
