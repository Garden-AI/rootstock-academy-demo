# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "academy-py>=0.5",
#     "rootstock>=1.2",  # 1.2's parallel prewarm matters on cold Lustre
#     "ase>=3.22",
#     "numpy",
#     "rich>=13",  # live TUI (--no-tui for plain logs)
# ]
# ///
"""Academy x Rootstock: a multi-MLIP query-by-committee campaign.

Showcase of an Academy agent system (https://docs.academy-agents.org) driving
several MLIPs at once through Rootstock's pre-built environments. One agent
per MLIP holds a warm ``RootstockCalculator`` (each model in its own isolated
venv/subprocess, so mutually-incompatible MLIPs coexist); a Curator agent runs
an autonomous active-learning loop: perturb known phases of an alloy system,
fan candidates out to the committee concurrently, rank by force disagreement
(query-by-committee uncertainty), and adaptively steer proposals toward the
regions where the models disagree most. The output is the ranked set of
structures a fine-tuning campaign would label first.

Seed structures come from the Materials Project (default: the stable Cu-Au
phases — fcc Cu/Au and the L1_2/L1_0 intermetallics), fetched once by
fetch_mp_structures.py into mp_seeds.extxyz; this script reads that file and
needs no network. Visualize the output with visualize_committee.py.

Run on a cluster compute node (see academy_committee_delta.sbatch /
academy_committee_sophia.pbs):

    uv run academy_mlip_committee.py --cluster delta

Try the wiring anywhere, no cluster required (EMT stand-ins for the MLIPs;
falls back to built-in Cu/Au seeds if mp_seeds.extxyz is absent):

    uv run academy_mlip_committee.py --mock
    uv run academy_mlip_committee.py --mock --builtin battery   # Li-ion seeds

Seeds with a mobile alkali (Li/Na/K) get battery-flavoured perturbation
arms: *delith* (remove mobile ions — the delithiation path), *hop* (kick
one mobile ion toward a neighbouring site — the migration precursor) and
*antisite* (mobile/transition-metal cation exchange — the LiFePO4/LiNiO2
defect); other seeds keep rattle/strain/vacancy/swap. Live progress renders
as a terminal dashboard (rich); pass --no-tui for plain line logging.

The member pool is chosen automatically from checkpoints the install's
manifest marks verified (in COMMITTEE_PREFERENCE order), so the demo tracks
whatever is actually healthy on the cluster. Override with --committee
id1,id2,...

Quorum start: cold model loads on Lustre can take ~20 min each, so the
campaign does not wait for the full pool — it starts as soon as --quorum
members are warm, and later members join the committee mid-campaign (the
report records the round each member joined). Members that fail or time out
are dropped; the campaign only aborts below 2 members.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from academy.agent import Agent, action, loop
from academy.exchange import LocalExchangeFactory
from academy.handle import Handle
from academy.logging.recommended import recommended_logging
from academy.manager import Manager
from ase import Atoms
from ase.build import bulk

# Candidate committee members, most-preferred first: one checkpoint per model
# family, spanning distinct training lineages so disagreement is meaningful.
# The actual committee is the first N of these that the target install's
# manifest marks verified.
COMMITTEE_PREFERENCE = [
    "mace-mp-0-medium",
    "tensornet-matpes-pbe-2025-2",
    "orb-v3-conservative-inf-omat",
    "sevennet-omat",
    "sevennet-mf-ompa",
    "chgnet-default",
    # Distinct lineages verified on Polaris (sevennet/mattersim are not there):
    "grace-2l-smax-omat-large",
    "pet-omatpes-l",
    "mattersim-v1-0-0-5m",
    # uma-s-1p1 removed 2026-07-30: its 14 GB checkpoint cache blows cold-node
    # warm-up budgets until rootstock prewarms weights (issues #177/#178).
    # Re-add once that ships — it's a strong committee member on warm nodes.
]


# Mobile ions that turn a seed into a "battery" seed (delith/hop/antisite
# arms), and the transition metals they exchange with in the antisite arm.
MOBILE_SPECIES = (3, 11, 19, 12)  # Li, Na, K, Mg — first one present wins
TRANSITION_METALS = frozenset(list(range(21, 31)) + list(range(39, 49))
                              + list(range(72, 81)))

# --------------------------------------------------------------------------
# Structures cross the agent boundary as plain dicts of lists so the demo
# works unchanged over any Academy exchange (local, Redis, ...), not just
# same-process handles.
# --------------------------------------------------------------------------

def atoms_to_payload(atoms: Atoms) -> dict:
    return {
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell[:].tolist(),
        "pbc": atoms.pbc.tolist(),
        # Provenance (e.g. mp_id) rides along into reports and output files.
        "info": {k: v for k, v in atoms.info.items()
                 if isinstance(v, (str, int, float, bool))},
    }


def payload_to_atoms(payload: dict) -> Atoms:
    atoms = Atoms(
        numbers=payload["numbers"],
        positions=payload["positions"],
        cell=payload["cell"],
        pbc=payload["pbc"],
    )
    atoms.info.update(payload.get("info", {}))
    return atoms


def supercell_to(atoms: Atoms, min_atoms: int) -> Atoms:
    """Replicate along the shortest lattice vector until >= min_atoms.

    Keeps cells roughly equiaxed without cubing the atom count (a 28-atom
    olivine cell becomes 56 atoms, not 224).
    """
    reps = np.ones(3, dtype=int)
    while len(atoms) * int(np.prod(reps)) < min_atoms:
        lengths = atoms.cell.lengths() * reps
        reps[int(np.argmin(lengths))] += 1
    return atoms * tuple(int(r) for r in reps)


def load_seeds(path: Path, min_atoms: int = 32) -> dict[str, Atoms]:
    """Read seed frames (from fetch_*_seeds.py) and supercell them."""
    from ase.io import read as ase_read

    seeds: dict[str, Atoms] = {}
    for atoms in ase_read(path, index=":"):
        name = str(atoms.info.get("seed_name", atoms.get_chemical_formula()))
        base, n = name, 2
        while name in seeds:
            name, n = f"{base}-{n}", n + 1
        seeds[name] = supercell_to(atoms, min_atoms)
    return seeds


def builtin_mock_seeds(min_atoms: int = 32) -> dict[str, Atoms]:
    """Cu/Au stand-in seeds so --mock runs without a fetched mp_seeds file."""
    a = 3.75
    cu3au = Atoms("AuCu3", cell=[a, a, a], pbc=True,
                  scaled_positions=[(0, 0, 0), (0, .5, .5),
                                    (.5, 0, .5), (.5, .5, 0)])
    cu3au.info["seed_name"] = "Cu3Au"
    frames = {
        "Cu": bulk("Cu", "fcc", a=3.615, cubic=True),
        "Au": bulk("Au", "fcc", a=4.078, cubic=True),
        "Cu3Au": cu3au,
    }
    return {name: supercell_to(atoms, min_atoms) for name, atoms in frames.items()}


def builtin_battery_seeds(min_atoms: int = 32) -> dict[str, Atoms]:
    """Textbook Li-ion phases from literature lattice parameters, so
    ``--mock --builtin battery`` runs without a Materials Project fetch.
    (The real campaign uses fetch_battery_seeds.py -> battery_seeds.extxyz.)"""
    from ase.spacegroup import crystal

    frames: dict[str, Atoms] = {}
    # Layered O3 cathodes, R-3m: Li 3a, TM 3b, O 6c.
    for tm, a, c, z_o in (("Co", 2.816, 14.05, 0.2395), ("Ni", 2.878, 14.19, 0.2415)):
        frames[f"Li{tm}O2"] = crystal(
            ["Li", tm, "O"], [(0, 0, 0), (0, 0, 0.5), (0, 0, z_o)],
            spacegroup=166, cellpar=[a, a, c, 90, 90, 120])
    # Spinel LiMn2O4, Fd-3m (origin choice 2): Li 8a, Mn 16d, O 32e.
    frames["LiMn2O4"] = crystal(
        ["Li", "Mn", "O"], [(0.125, 0.125, 0.125), (0.5, 0.5, 0.5), (0.263, 0.263, 0.263)],
        spacegroup=227, cellpar=[8.24] * 3 + [90] * 3, setting=2)
    # Olivine LiFePO4, Pnma.
    frames["LiFePO4"] = crystal(
        ["Li", "Fe", "P", "O", "O", "O"],
        [(0, 0, 0), (0.2822, 0.25, 0.9748), (0.0949, 0.25, 0.4182),
         (0.0968, 0.25, 0.7428), (0.4573, 0.25, 0.2059), (0.1655, 0.0466, 0.2851)],
        spacegroup=62, cellpar=[10.33, 6.01, 4.69, 90, 90, 90])
    # Antifluorite Li2O, Fm-3m: Li 8c, O 4a.
    frames["Li2O"] = crystal(["Li", "O"], [(0.25, 0.25, 0.25), (0, 0, 0)],
                             spacegroup=225, cellpar=[4.61] * 3 + [90] * 3)
    frames["Li"] = bulk("Li", "bcc", a=3.51, cubic=True)
    # Graphite anode, P6_3/mmc: C 2b + 2c.
    frames["C"] = crystal(["C", "C"], [(0, 0, 0.25), (1 / 3, 2 / 3, 0.25)],
                          spacegroup=194, cellpar=[2.46, 2.46, 6.71, 90, 90, 120])
    for name, atoms in frames.items():
        atoms.info["seed_name"] = name
    return {name: supercell_to(atoms, min_atoms) for name, atoms in frames.items()}


class _MockMLIP:
    """EMT with a per-member bias, standing in for a real MLIP in --mock runs.

    Each mock member scales forces and adds seeded noise, so the committee
    disagrees more where forces are large — enough signal for the Curator's
    adaptive loop to visibly chase disorder.
    """

    # Elements ASE's EMT parametrises; anything else falls back to a soft
    # Morse pair potential so battery-style seeds (Li/Co/O/P...) still run
    # in --mock (a hard r^-12 wall explodes on rattled 1.5 A P-O bonds).
    _EMT_SPECIES = {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "H", "C", "N", "O"}

    def __init__(self, member_index: int):
        from ase.calculators.emt import EMT
        from ase.calculators.morse import MorsePotential

        self._emt = EMT()
        self._lj = MorsePotential(epsilon=0.25, r0=2.0, rho0=3.5,
                                  rcut1=2.5, rcut2=3.2)
        self._scale = 1.0 + 0.015 * member_index
        self._noise = 0.01 + 0.015 * member_index
        self._seed = 1000 + member_index

    def evaluate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        symbols = set(atoms.get_chemical_symbols())
        atoms.calc = self._emt if symbols <= self._EMT_SPECIES else self._lj
        energy = atoms.get_potential_energy() * self._scale
        forces = atoms.get_forces() * self._scale
        structure_key = zlib.crc32(np.ascontiguousarray(atoms.positions).tobytes())
        rng = np.random.default_rng((self._seed, structure_key))
        forces = forces + rng.normal(0.0, self._noise, forces.shape)
        return float(energy), forces

    def close(self) -> None:
        pass


class _RootstockMLIP:
    """One warm RootstockCalculator, reused across every structure we score."""

    def __init__(self, checkpoint: str, root: str, device: str, timeout: float):
        from rootstock import RootstockCalculator

        self._calc = RootstockCalculator(
            checkpoint=checkpoint, root=root, device=device, timeout=timeout
        )

    def evaluate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        atoms.calc = self._calc
        return float(atoms.get_potential_energy()), atoms.get_forces()

    def close(self) -> None:
        self._calc.close()


class MLIPCommitteeMember(Agent):
    """Wraps one MLIP behind an Academy action interface.

    The expensive part — spawning the env's worker subprocess and loading the
    model — happens once, in ``agent_on_startup``; every ``evaluate_batch``
    after that hits a warm calculator. Blocking calculator calls run via
    ``agent_run_sync`` so members evaluate genuinely in parallel.
    """

    def __init__(self, name: str, checkpoint: str, root: str | None,
                 device: str, timeout: float = 1800.0,
                 mock_index: int | None = None):
        super().__init__()
        self._name = name
        self._checkpoint = checkpoint
        self._root = root
        self._device = device
        self._timeout = timeout
        self._mock_index = mock_index
        self._mlip: _MockMLIP | _RootstockMLIP | None = None
        self._warmup_s: float | None = None
        self._n_evaluated = 0

    def _build_and_warm(self):
        if self._mock_index is not None:
            # Stagger mock warm-ups so quorum start and mid-campaign joins
            # are exercised (and demo-able) without a cluster.
            time.sleep(4.0 * self._mock_index)
            mlip = _MockMLIP(self._mock_index)
        else:
            mlip = _RootstockMLIP(self._checkpoint, self._root, self._device,
                                  self._timeout)
        # Pay worker spawn + model load now, not inside the first campaign round.
        mlip.evaluate(bulk("Cu", "fcc", a=3.615))
        return mlip

    async def agent_on_startup(self) -> None:
        start = time.monotonic()
        self._mlip = await self.agent_run_sync(self._build_and_warm)
        self._warmup_s = time.monotonic() - start

    async def agent_on_shutdown(self) -> None:
        if self._mlip is not None:
            self._mlip.close()

    @action
    async def info(self) -> dict:
        # Also serves as the readiness barrier: actions aren't processed
        # until agent_on_startup completes, i.e. until the model is warm.
        return {
            "name": self._name,
            "checkpoint": self._checkpoint,
            "warmup_s": round(self._warmup_s, 1),
            "n_evaluated": self._n_evaluated,
        }

    def _evaluate(self, payloads: list[dict]) -> list[dict]:
        results = []
        for payload in payloads:
            energy, forces = self._mlip.evaluate(payload_to_atoms(payload))
            results.append({"energy": energy, "forces": forces.tolist()})
        self._n_evaluated += len(payloads)
        return results

    @action
    async def evaluate_batch(self, payloads: list[dict]) -> list[dict]:
        return await self.agent_run_sync(self._evaluate, payloads)


# --------------------------------------------------------------------------
# Curator: the autonomous active-learning loop.
# --------------------------------------------------------------------------

@dataclass
class CampaignConfig:
    rounds: int = 4
    batch_size: int = 12
    top_k: int = 10
    seed: int = 7
    # Pause between rounds; lets slow-warming members join mid-campaign even
    # when evaluation itself is fast (also paces the mock demo).
    round_pause: float = 0.0


@dataclass
class _Arm:
    """One (seed structure, perturbation family) proposal arm."""

    seed_name: str
    transform: str
    magnitude: float
    cap: float
    scores: list[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


class Curator(Agent):
    """Proposes structures, queries the committee, and chases disagreement.

    Query-by-committee: a structure's score is the largest per-atom standard
    deviation of forces across the committee (eV/Å). High score = the models
    genuinely disagree there = the most informative structure to label with
    DFT and fold into fine-tuning.
    """

    def __init__(self, pool: dict[str, Handle], active: set[str],
                 seed_payloads: dict[str, dict], config: CampaignConfig):
        super().__init__()
        # `pool` holds handles to every launched member, warm or not; only
        # names in `_active` are queried. Stragglers are added via the
        # activate_member action as their models finish loading.
        self._pool = dict(pool)
        self._active = set(active)
        self._joined_round: dict[str, int] = {name: 0 for name in active}
        self._cfg = config
        self._rng = np.random.default_rng(config.seed)
        self._seeds = {name: payload_to_atoms(p)
                       for name, p in seed_payloads.items()}
        self._arms = self._make_arms()
        self._candidates: list[dict] = []
        self._strikes: dict[str, int] = {name: 0 for name in pool}
        self._round = 0
        self._status_note = "starting campaign"
        self._done = False
        self._report: dict | None = None
        # Live telemetry for the driver's dashboard / event log.
        self._events: list[dict] = []
        self._t0 = time.monotonic()
        self._member_stats: dict[str, dict] = {
            name: {"n_evals": 0, "last_batch_s": None, "state": "seated"}
            for name in active}
        self._best_per_round: list[float] = []
        self._emit("campaign_start", members=sorted(active),
                   n_arms=len(self._arms), n_seeds=len(self._seeds))

    def _emit(self, kind: str, **data) -> None:
        self._events.append({"t": round(time.monotonic() - self._t0, 2),
                             "round": self._round, "kind": kind, **data})

    # -- proposal ----------------------------------------------------------

    @staticmethod
    def _mobile_species(atoms: Atoms) -> int | None:
        present = set(atoms.numbers.tolist())
        return next((z for z in MOBILE_SPECIES if z in present), None)

    def _make_arms(self) -> list[_Arm]:
        arms = []
        for seed_name, atoms in self._seeds.items():
            arms.append(_Arm(seed_name, "rattle", magnitude=0.05, cap=0.35))
            arms.append(_Arm(seed_name, "strain", magnitude=0.02, cap=0.08))
            mobile = self._mobile_species(atoms)
            if mobile is not None:
                # Battery seed: the physics that matters is what happens to
                # the mobile ion.  Magnitudes: fraction of mobile ions
                # removed; hop displacement in Å; fraction of mobile ions
                # exchanged with transition-metal sites.
                arms.append(_Arm(seed_name, "delith", magnitude=0.1, cap=0.5))
                arms.append(_Arm(seed_name, "hop", magnitude=0.5, cap=1.6))
                if TRANSITION_METALS & set(atoms.numbers.tolist()):
                    arms.append(_Arm(seed_name, "antisite", magnitude=0.05, cap=0.3))
            else:
                arms.append(_Arm(seed_name, "vacancy", magnitude=1.0, cap=3.0))
                if len(set(atoms.numbers)) >= 2:
                    # Antisite disorder — only meaningful for intermetallics.
                    arms.append(_Arm(seed_name, "swap", magnitude=0.1, cap=0.5))
        return arms

    def _apply_transform(self, arm: _Arm) -> Atoms:
        atoms = self._seeds[arm.seed_name].copy()
        rng = self._rng
        if arm.transform == "rattle":
            atoms.positions += rng.normal(0.0, arm.magnitude, atoms.positions.shape)
        elif arm.transform == "strain":
            strain = rng.uniform(-arm.magnitude, arm.magnitude, (3, 3))
            strain = (strain + strain.T) / 2 + np.eye(3)
            atoms.set_cell(atoms.cell[:] @ strain, scale_atoms=True)
            atoms.positions += rng.normal(0.0, 0.02, atoms.positions.shape)
        elif arm.transform == "vacancy":
            n_vac = max(1, int(round(arm.magnitude)))
            keep = rng.choice(len(atoms), size=len(atoms) - n_vac, replace=False)
            atoms = atoms[np.sort(keep)]
            atoms.positions += rng.normal(0.0, 0.03, atoms.positions.shape)
        elif arm.transform == "delith":
            # Remove a fraction of the mobile ions (delithiation path). The
            # remaining lattice gets a small rattle so forces are non-trivial.
            mobile = self._mobile_species(atoms)
            idx = np.flatnonzero(atoms.numbers == mobile)
            n_rm = max(1, int(round(arm.magnitude * len(idx))))
            drop = rng.choice(idx, size=min(n_rm, len(idx)), replace=False)
            keep = np.setdiff1d(np.arange(len(atoms)), drop)
            atoms = atoms[keep]
            atoms.positions += rng.normal(0.0, 0.03, atoms.positions.shape)
        elif arm.transform == "hop":
            # Kick one mobile ion toward its nearest same-species neighbour
            # (a migration-path midpoint at magnitude ~ half the hop length).
            mobile = self._mobile_species(atoms)
            idx = np.flatnonzero(atoms.numbers == mobile)
            i = int(rng.choice(idx))
            d = atoms.get_distances(i, idx, mic=True, vector=True)
            norms = np.linalg.norm(d, axis=1)
            norms[idx == i] = np.inf
            j = int(np.argmin(norms))
            direction = d[j] / norms[j] if np.isfinite(norms[j]) else rng.normal(size=3)
            atoms.positions[i] += direction / np.linalg.norm(direction) * arm.magnitude
            atoms.positions += rng.normal(0.0, 0.03, atoms.positions.shape)
        elif arm.transform in ("swap", "antisite"):
            # Composition-preserving antisite pairs: exchange the species of
            # k A-sites with k B-sites (order-disorder along the L1_2/L1_0
            # story; Li/TM cation mixing for battery seeds), plus a small
            # rattle to break the ideal-lattice symmetry.
            numbers = atoms.numbers.copy()
            if arm.transform == "antisite":
                species_a = self._mobile_species(atoms)
                tm = sorted(TRANSITION_METALS & set(numbers.tolist()))
                # Most abundant TM is the antisite partner (Fe in LiFePO4).
                species_b = max(tm, key=lambda z: int((numbers == z).sum()))
            else:
                species_a, species_b = np.unique(numbers)[:2]
            idx_a = np.flatnonzero(numbers == species_a)
            idx_b = np.flatnonzero(numbers == species_b)
            k = max(1, int(round(arm.magnitude * len(atoms) / 2)))
            k = min(k, len(idx_a), len(idx_b))
            swap_a = rng.choice(idx_a, size=k, replace=False)
            swap_b = rng.choice(idx_b, size=k, replace=False)
            numbers[swap_a], numbers[swap_b] = species_b, species_a
            atoms.set_atomic_numbers(numbers)
            atoms.positions += rng.normal(0.0, 0.03, atoms.positions.shape)
        return atoms

    def _propose(self, round_no: int) -> list[dict]:
        batch_size = self._cfg.batch_size
        if round_no == 0:
            # Survey round: cover every arm evenly.
            picks = [self._arms[i % len(self._arms)] for i in range(batch_size)]
        else:
            # Exploit: sample arms proportional to observed disagreement,
            # with a floor so no arm is starved of exploration.
            weights = np.array([arm.mean_score + 0.02 for arm in self._arms])
            picks = list(self._rng.choice(self._arms, size=batch_size,
                                          p=weights / weights.sum()))
        batch = []
        for i, arm in enumerate(picks):
            batch.append({
                "id": f"r{round_no}-{i:02d}",
                "round": round_no,
                "seed": arm.seed_name,
                "transform": arm.transform,
                "magnitude": round(arm.magnitude, 3),
                "payload": atoms_to_payload(self._apply_transform(arm)),
                "_arm": arm,
            })
        return batch

    # -- scoring -----------------------------------------------------------

    @staticmethod
    def _score(member_results: dict[str, dict]) -> tuple[float, float]:
        forces = np.array([r["forces"] for r in member_results.values()])
        energies = np.array([r["energy"] for r in member_results.values()])
        n_atoms = forces.shape[1]
        per_atom = np.linalg.norm(forces.std(axis=0), axis=1)
        e_spread = float(energies.max() - energies.min()) / n_atoms
        return float(per_atom.max()), e_spread

    # -- the autonomous campaign loop ---------------------------------------

    @action
    async def activate_member(self, name: str) -> None:
        """A late-warming pool member is ready: seat it on the committee."""
        if name in self._pool and name not in self._active:
            self._active.add(name)
            self._joined_round[name] = self._round
            self._member_stats[name] = {"n_evals": 0, "last_batch_s": None,
                                        "state": "seated"}
            self._status_note = f"{name} joined the committee"
            self._emit("member_joined", member=name)

    @loop
    async def campaign(self, shutdown: asyncio.Event) -> None:
        for round_no in range(self._cfg.rounds):
            if shutdown.is_set():
                return
            self._round = round_no + 1
            batch = self._propose(round_no)
            payloads = [c["payload"] for c in batch]

            live = {name: self._pool[name] for name in sorted(self._active)
                    if self._strikes[name] < 2}
            if len(live) < 2:
                self._status_note = "fewer than 2 healthy committee members; aborting"
                break
            for name in live:
                self._member_stats[name]["state"] = "scoring"
            self._emit("round_start", n=len(batch), members=sorted(live))

            async def timed(name: str, handle: Handle):
                t = time.monotonic()
                res = await handle.evaluate_batch(payloads)
                return name, res, time.monotonic() - t

            results = await asyncio.gather(
                *(timed(name, handle) for name, handle in live.items()),
                return_exceptions=True,
            )

            per_member: dict[str, list[dict]] = {}
            for name, result in zip(live, results):
                if isinstance(result, BaseException):
                    self._strikes[name] += 1
                    self._status_note = f"member {name} failed: {result!r}"
                    self._member_stats[name]["state"] = (
                        "dropped" if self._strikes[name] >= 2 else "strike")
                    self._emit("member_failed", member=name,
                               strikes=self._strikes[name], error=repr(result)[:120])
                else:
                    _, res, dt = result
                    self._strikes[name] = 0
                    per_member[name] = res
                    st = self._member_stats[name]
                    st.update(n_evals=st["n_evals"] + len(payloads),
                              last_batch_s=round(dt, 2), state="seated")

            best_this_round = None
            for i, cand in enumerate(batch):
                member_results = {name: res[i] for name, res in per_member.items()}
                if len(member_results) < 2:
                    continue
                score, e_spread = self._score(member_results)
                arm = cand.pop("_arm")
                arm.scores.append(score)
                cand["qbc_force_std"] = round(score, 4)
                cand["energy_spread_per_atom"] = round(e_spread, 4)
                cand["energies"] = {n: round(r["energy"], 4)
                                    for n, r in member_results.items()}
                # Per-member forces ride along (kept only for the selected
                # top-k in the report) so report_html.py can draw each
                # model's force arrows on the same atom.
                cand["member_forces"] = {
                    n: np.round(np.asarray(r["forces"]), 4).tolist()
                    for n, r in member_results.items()}
                self._candidates.append(cand)
                if best_this_round is None or score > best_this_round["qbc_force_std"]:
                    best_this_round = cand

            # Escalate the winning arm: push harder where models disagree.
            if best_this_round is not None:
                for arm in self._arms:
                    if (arm.seed_name == best_this_round["seed"]
                            and arm.transform == best_this_round["transform"]):
                        arm.magnitude = min(arm.magnitude * 1.35, arm.cap)
                        escalated_to = arm.magnitude
                self._status_note = (
                    f"best {best_this_round['qbc_force_std']:.3f} eV/A from "
                    f"{best_this_round['seed']}/{best_this_round['transform']}"
                    f"@{best_this_round['magnitude']}"
                )
                self._best_per_round.append(best_this_round["qbc_force_std"])
                self._emit("round_done", best=best_this_round["qbc_force_std"],
                           seed=best_this_round["seed"],
                           transform=best_this_round["transform"],
                           magnitude=best_this_round["magnitude"],
                           escalated_to=round(escalated_to, 3),
                           evaluated=len(self._candidates))

            if self._cfg.round_pause:
                await asyncio.sleep(self._cfg.round_pause)

        ranked = sorted(self._candidates, key=lambda c: c["qbc_force_std"],
                        reverse=True)
        history_keys = ("id", "round", "seed", "transform", "magnitude",
                        "qbc_force_std", "energy_spread_per_atom", "energies")
        self._report = {
            # name -> round it joined (0 = founding quorum member).
            "committee": {n: self._joined_round[n] for n in sorted(self._active)},
            "never_joined": sorted(set(self._pool) - self._active),
            "dropped_members": [n for n, s in self._strikes.items() if s >= 2],
            "rounds_completed": self._round,
            "seeds": {name: dict(atoms.info, natoms=len(atoms))
                      for name, atoms in self._seeds.items()},
            # Every scored candidate, payload-free — for visualization.
            "history": [{k: c[k] for k in history_keys}
                        for c in self._candidates],
            "n_evaluated": len(self._candidates),
            "arms": [{
                "seed": arm.seed_name, "transform": arm.transform,
                "final_magnitude": round(arm.magnitude, 3),
                "mean_qbc": round(arm.mean_score, 4),
                "n": len(arm.scores),
            } for arm in self._arms],
            "selected": ranked[: self._cfg.top_k],
        }
        # Only the selected structures carry their (bulky) member forces.
        selected_ids = {c["id"] for c in self._report["selected"]}
        for cand in self._candidates:
            if cand["id"] not in selected_ids:
                cand.pop("member_forces", None)
        self._emit("campaign_done", evaluated=len(self._candidates),
                   top=[c["id"] for c in ranked[:3]])
        self._done = True

    @action
    async def status(self) -> dict:
        arms = sorted(self._arms, key=lambda a: a.mean_score, reverse=True)
        return {"round": self._round, "rounds": self._cfg.rounds,
                "members": len(self._active),
                "evaluated": len(self._candidates), "note": self._status_note,
                "done": self._done,
                "member_stats": self._member_stats,
                "best_per_round": self._best_per_round,
                "arms": [{"seed": a.seed_name, "transform": a.transform,
                          "mean_qbc": round(a.mean_score, 4), "n": len(a.scores),
                          "magnitude": round(a.magnitude, 3)} for a in arms[:8]]}

    @action
    async def events(self, since: int = 0) -> list[dict]:
        """Event stream (append-only); pass the count you've already seen."""
        return self._events[since:]

    @action
    async def report(self) -> dict:
        if self._report is None:
            raise RuntimeError("campaign still running")
        return self._report


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def resolve_committee(root: Path, requested: list[str] | None, size: int) -> list[str]:
    """Pick the committee from checkpoints the manifest says are verified."""
    from rootstock.manifest import is_verified, load_manifest

    manifest = load_manifest(root)
    if manifest is None:
        raise SystemExit(f"no rootstock manifest at {root} — is this an install root?")
    verified: dict[str, str] = {}  # checkpoint id -> env name
    for env_name, env in manifest.environments.items():
        for ckpt_id, ckpt in env.checkpoints.items():
            if is_verified(env, ckpt):
                verified[ckpt_id] = env_name

    if requested:
        for ckpt_id in requested:
            if ckpt_id not in verified:
                print(f"WARNING: {ckpt_id} is not marked verified in the manifest; "
                      "trying it anyway.")
        return requested

    committee, used_envs = [], set()
    for ckpt_id in COMMITTEE_PREFERENCE:
        env = verified.get(ckpt_id)
        if env is not None and env not in used_envs:
            committee.append(ckpt_id)
            used_envs.add(env)
        if len(committee) == size:
            break
    if len(committee) < 2:
        raise SystemExit(
            "need at least 2 verified committee checkpoints; manifest has: "
            + (", ".join(sorted(verified)) or "none")
        )
    return committee


def print_report(report: dict, outdir: Path) -> None:
    from ase.io import write as ase_write

    print(f"\n=== campaign report: {report['n_evaluated']} structures, "
          f"{report['rounds_completed']} rounds ===")
    members = ", ".join(
        f"{name} (joined r{joined})" if joined else name
        for name, joined in report["committee"].items()
    )
    print(f"final committee: {members}")
    if report["never_joined"]:
        print(f"never joined: {', '.join(report['never_joined'])}")
    print("\narm ranking (where does the committee disagree?):")
    for arm in sorted(report["arms"], key=lambda a: a["mean_qbc"], reverse=True):
        print(f"  {arm['seed']:>8s}/{arm['transform']:<8s} "
              f"mean QbC {arm['mean_qbc']:.3f} eV/A over {arm['n']:2d} structs "
              f"(escalated to {arm['final_magnitude']})")

    print(f"\ntop {len(report['selected'])} structures to label first:")
    print(f"  {'id':>8s} {'seed':>8s} {'transform':>9s} {'mag':>6s} "
          f"{'QbC eV/A':>9s} {'dE/atom eV':>11s}")
    selected_atoms = []
    for cand in report["selected"]:
        print(f"  {cand['id']:>8s} {cand['seed']:>8s} {cand['transform']:>9s} "
              f"{cand['magnitude']:>6} {cand['qbc_force_std']:>9.3f} "
              f"{cand['energy_spread_per_atom']:>11.4f}")
        atoms = payload_to_atoms(cand["payload"])
        atoms.info.update({k: cand[k] for k in
                           ("id", "seed", "transform", "magnitude",
                            "qbc_force_std", "energy_spread_per_atom")})
        selected_atoms.append(atoms)

    outdir.mkdir(parents=True, exist_ok=True)
    ase_write(outdir / "selected_structures.extxyz", selected_atoms)
    (outdir / "committee_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {outdir}/selected_structures.extxyz and committee_report.json")


class Dashboard:
    """Live terminal view of the committee: who is warm, who is scoring, who
    struck out; the campaign's disagreement trend; the arms being chased.
    Falls back to plain line logging (SLURM .out friendly) with --no-tui or
    when stdout is not a terminal."""

    STATE_STYLE = {"warming": ("◐", "yellow"), "seated": ("●", "green"),
                   "scoring": ("◉", "bright_green"), "strike": ("◑", "red"),
                   "dropped": ("✕", "red"), "never joined": ("○", "dim")}
    SPARK = "▁▂▃▄▅▆▇█"

    def __init__(self, title: str, pool: list[str], quorum: int, live: bool,
                 svg_out: Path | None = None):
        self.title, self.pool, self.quorum, self.live = title, pool, quorum, live
        self.svg_out = svg_out
        self.t0 = time.monotonic()
        self.member: dict[str, dict] = {
            n: {"state": "warming", "warm_s": None, "joined": None,
                "n_evals": 0, "last_batch_s": None, "strikes": 0}
            for n in pool}
        self.status: dict = {}
        self.events: list[dict] = []
        self._rich = None
        if live:
            from rich.console import Console
            from rich.live import Live
            self._console = Console(record=True)
            self._rich = Live(self._render(), console=self._console,
                              refresh_per_second=4, screen=False)
            self._rich.start()

    # -- updates -----------------------------------------------------------
    def warm(self, name: str, warm_s: float, joined_round: int | None) -> None:
        m = self.member[name]
        m.update(state="seated", warm_s=warm_s, joined=joined_round)
        self.log(f"{name} warm in {warm_s:.0f}s"
                 + (f" — seated for round {joined_round + 1}" if joined_round else ""))

    def dropped(self, name: str, why: str) -> None:
        self.member[name].update(state="dropped")
        self.log(f"{name} DROPPED at startup: {why}")

    def update_status(self, status: dict) -> None:
        self.status = status
        for name, st in status.get("member_stats", {}).items():
            m = self.member.get(name)
            if m is None:
                continue
            m.update(n_evals=st["n_evals"], last_batch_s=st["last_batch_s"])
            if m["state"] != "dropped":
                m["state"] = st["state"]
        if self._rich:
            self._rich.update(self._render())

    def add_events(self, events: list[dict]) -> None:
        for ev in events:
            self.events.append(ev)
            self.log(self._describe(ev), quiet=True)

    def finish(self) -> None:
        for m in self.member.values():
            if m["state"] == "warming":
                m["state"] = "never joined"
        if self._rich:
            self._rich.update(self._render())
            self._rich.stop()
            if self.svg_out is not None:
                # Final board as a shareable image (Rich's own SVG export).
                self._console.export_text(clear=True)  # drop Live's frames
                self._console.print(self._render())
                self._console.save_svg(str(self.svg_out), title=self.title)

    def log(self, msg: str, quiet: bool = False) -> None:
        if not self.live:
            print(f"[t+{time.monotonic() - self.t0:5.0f}s] {msg}", flush=True)
        elif not quiet:
            self.events.append({"t": time.monotonic() - self.t0, "kind": "driver",
                                "msg": msg})

    @staticmethod
    def _describe(ev: dict) -> str:
        k = ev["kind"]
        if k == "campaign_start":
            return f"campaign start: {len(ev['members'])} members, {ev['n_seeds']} seeds, {ev['n_arms']} arms"
        if k == "round_start":
            return f"round {ev['round']}: {ev['n']} structures → {len(ev['members'])} members"
        if k == "round_done":
            return (f"round {ev['round']} best σF {ev['best']:.3f} eV/Å "
                    f"{ev['seed']}/{ev['transform']}@{ev['magnitude']} → escalate to {ev['escalated_to']}")
        if k == "member_joined":
            return f"{ev['member']} joined the committee (round {ev['round']})"
        if k == "member_failed":
            return f"{ev['member']} strike {ev['strikes']}: {ev['error']}"
        if k == "campaign_done":
            return f"campaign done: {ev['evaluated']} scored; top {', '.join(ev['top'])}"
        return ev.get("msg", k)

    # -- rendering ---------------------------------------------------------
    def _render(self):
        from rich import box
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        st = self.status
        elapsed = time.monotonic() - self.t0
        head = Text.assemble(
            (f" {self.title} ", "bold"),
            (f"  t+{elapsed // 60:02.0f}:{elapsed % 60:02.0f}", "dim"),
            (f"   round {st.get('round', 0)}/{st.get('rounds', '?')}", "cyan"),
            (f"   {st.get('evaluated', 0)} scored", "dim"),
            (f"   quorum {self.quorum}", "dim"))

        mt = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False,
                   header_style="dim", show_edge=False)
        for col, j in (("member", "left"), ("state", "left"), ("warm-up", "right"),
                       ("evals", "right"), ("last batch", "right"), ("strikes", "right")):
            mt.add_column(col, justify=j)
        for name in self.pool:
            m = self.member[name]
            glyph, style = self.STATE_STYLE.get(m["state"], ("?", ""))
            state = m["state"]
            if m["joined"]:
                state += f" (r{m['joined'] + 1})"
            warm = "…" if m["warm_s"] is None else f"{m['warm_s']:.0f}s"
            lb = "–" if m["last_batch_s"] is None else f"{m['last_batch_s']:.1f}s"
            mt.add_row(Text(name, style=style if m["state"] in ("dropped", "never joined") else ""),
                       Text(f"{glyph} {state}", style=style), warm,
                       str(m["n_evals"]), lb, str(m["strikes"]))

        best = st.get("best_per_round", [])
        if best:
            hi = max(best) or 1.0
            spark = "".join(self.SPARK[min(7, int(b / hi * 7.999))] for b in best)
            trend = Text.assemble(("σ(F) best/round  ", "dim"), (spark, "cyan"),
                                  (f"  {best[-1]:.3f} eV/Å", "bold"))
        else:
            trend = Text("σ(F) best/round  waiting for first round…", style="dim")
        note = Text(st.get("note", ""), style="dim")

        at = Table(box=None, expand=True, pad_edge=False, header_style="dim")
        at.add_column("arm"); at.add_column("mean σF", justify="right")
        at.add_column("", ratio=2); at.add_column("n", justify="right")
        at.add_column("→ magnitude", justify="right")
        arms = st.get("arms", [])
        amax = max((a["mean_qbc"] for a in arms), default=0) or 1.0
        for a in arms[:8]:
            bar = "█" * int(round(18 * a["mean_qbc"] / amax))
            at.add_row(f"{a['seed']} · {a['transform']}", f"{a['mean_qbc']:.3f}",
                       Text(bar, style="blue"), str(a["n"]), str(a["magnitude"]))

        ev = Table(box=None, expand=True, pad_edge=False, show_header=False)
        ev.add_column("t", justify="right", style="dim", width=7)
        ev.add_column("event")
        for e in self.events[-8:]:
            ev.add_row(f"{e['t']:5.0f}s", self._describe(e) if "kind" in e else e["msg"])

        return Group(
            Panel(head, box=box.HEAVY, padding=(0, 0)),
            Panel(mt, title="committee", title_align="left", box=box.ROUNDED),
            Panel(Group(trend, note), title="campaign", title_align="left", box=box.ROUNDED),
            Panel(at, title="arms (where the committee disagrees)", title_align="left", box=box.ROUNDED),
            Panel(ev, title="events", title_align="left", box=box.ROUNDED),
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cluster", default="delta",
                        help="rootstock cluster name (default: delta)")
    parser.add_argument("--root", default=None,
                        help="explicit install root (overrides --cluster)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="rootstock worker timeout in seconds; covers the "
                             "cold-cache model load, which contends for Lustre "
                             "bandwidth when members warm up in parallel "
                             "(default: 1800)")
    parser.add_argument("--committee", default=None,
                        help="comma-separated checkpoint ids (default: auto "
                             "from the manifest's verified set)")
    parser.add_argument("--pool-size", type=int, default=6,
                        help="members to launch (default: 6)")
    parser.add_argument("--quorum", type=int, default=3,
                        help="warm members needed to start the campaign; "
                             "later members join mid-flight (default: 3)")
    parser.add_argument("--round-pause", type=float, default=0.0,
                        help="seconds to pause between rounds, giving "
                             "slow-warming members rounds to join (default: 0)")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=Path, default=Path("committee_results"))
    parser.add_argument("--seeds", type=Path, default=None,
                        help="extxyz of seed structures from "
                             "fetch_mp_structures.py (default: mp_seeds.extxyz "
                             "next to this script)")
    parser.add_argument("--mock", action="store_true",
                        help="EMT/LJ stand-ins instead of Rootstock (runs anywhere)")
    parser.add_argument("--builtin", choices=("cuau", "battery"), default=None,
                        help="use built-in seeds instead of a seeds file")
    parser.add_argument("--no-tui", action="store_true",
                        help="plain line logging instead of the live dashboard "
                             "(automatic when stdout is not a terminal)")
    args = parser.parse_args()

    seeds_path = args.seeds or Path(__file__).parent / "mp_seeds.extxyz"
    if args.builtin == "battery":
        seeds = builtin_battery_seeds()
    elif args.builtin == "cuau":
        seeds = builtin_mock_seeds()
    elif seeds_path.exists():
        seeds = load_seeds(seeds_path)
    elif args.mock:
        print(f"no seeds file at {seeds_path}; using built-in Cu/Au mock seeds")
        seeds = builtin_mock_seeds()
    else:
        raise SystemExit(
            f"no seeds file at {seeds_path} — fetch one first (needs internet "
            "+ MP_API_KEY):  uv run fetch_mp_structures.py"
        )
    print("seeds: " + ", ".join(
        f"{name} ({atoms.info.get('mp_id', 'built-in')}, {len(atoms)} atoms)"
        for name, atoms in seeds.items()))

    args.quorum = max(2, args.quorum)
    if args.mock:
        root = None
        pool = [f"mock-emt-{i}" for i in range(args.pool_size)]
    else:
        from rootstock.clusters import get_cluster

        root = Path(args.root) if args.root else get_cluster(args.cluster).root
        requested = args.committee.split(",") if args.committee else None
        pool = resolve_committee(root, requested, args.pool_size)
    print(f"member pool: {', '.join(pool)}")

    import sys
    live_tui = not args.no_tui and sys.stdout.isatty()
    dash = Dashboard(f"Academy × Rootstock committee — "
                     f"{'mock' if args.mock else args.cluster}",
                     pool, args.quorum, live=live_tui,
                     svg_out=args.outdir / "committee_dashboard.svg")
    args.outdir.mkdir(parents=True, exist_ok=True)
    event_log = (args.outdir / "committee_events.jsonl").open("w")

    executor = ThreadPoolExecutor(max_workers=len(pool) + 4)
    campaign_done = False
    try:
      async with await Manager.from_exchange_factory(
          factory=LocalExchangeFactory(),
          executors=executor,
          log_config=recommended_logging(level="WARNING"),
      ) as manager:
        # Launch the whole pool at once; each member warms its model in
        # agent_on_startup, so the (expensive) model loads run in parallel.
        dash.log(f"launching {len(pool)} member agents (parallel model warm-up); "
                 f"campaign starts at quorum of {args.quorum}...")
        t0 = time.monotonic()
        handles: dict[str, Handle] = dict(zip(pool, await asyncio.gather(*(
            manager.launch(
                MLIPCommitteeMember,
                args=(ckpt, ckpt, str(root) if root else None, args.device,
                      args.timeout, i if args.mock else None),
                name=ckpt,
            )
            for i, ckpt in enumerate(pool)
        ))))

        # info() doubles as the per-member readiness signal (actions queue
        # until agent_on_startup completes). A member whose model fails or
        # times out is dropped, not fatal.
        info_tasks = {asyncio.create_task(handles[c].info()): c for c in pool}
        pending = set(info_tasks)

        current_round = 0

        def absorb(task) -> str | None:
            """Report one finished readiness task; member name if it warmed."""
            ckpt = info_tasks[task]
            exc = task.exception()
            if exc is not None:
                cause = exc.__cause__ or exc.__context__ or exc
                dash.dropped(ckpt, f"{cause} (full traceback in the agent error log)")
                return None
            dash.warm(ckpt, task.result()["warmup_s"], current_round or None)
            return ckpt

        active: set[str] = set()
        while pending and len(active) < args.quorum:
            ready, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
            active.update(filter(None, map(absorb, ready)))
        if len(active) < 2:
            raise SystemExit("fewer than 2 committee members started; aborting")
        dash.log(f"quorum of {len(active)} reached at t+{time.monotonic() - t0:.0f}s"
                 f" — campaign starting ({len(pending)} members still warming)")

        curator = await manager.launch(
            Curator,
            args=(handles, active,
                  {name: atoms_to_payload(atoms)
                   for name, atoms in seeds.items()},
                  CampaignConfig(rounds=args.rounds, batch_size=args.batch_size,
                                 top_k=args.top_k, seed=args.seed,
                                 round_pause=args.round_pause)),
            name="curator",
        )

        last_line, n_seen = None, 0
        poll = 1.0 if live_tui else 3.0
        while True:
            # Seat stragglers as they warm (asyncio.wait's timeout doubles
            # as the status-poll cadence).
            if pending:
                ready, pending = await asyncio.wait(
                    pending, timeout=poll, return_when=asyncio.FIRST_COMPLETED)
                for name in filter(None, map(absorb, ready)):
                    await curator.activate_member(name)
            else:
                await asyncio.sleep(poll)

            status = await curator.status()
            current_round = status["round"]
            new_events = await curator.events(n_seen)
            n_seen += len(new_events)
            for ev in new_events:
                event_log.write(json.dumps(ev) + "\n")
            event_log.flush()
            dash.add_events(new_events)
            dash.update_status(status)
            if not live_tui:
                line = (f"[round {status['round']}/{status['rounds']}, "
                        f"committee of {status['members']}] "
                        f"{status['evaluated']} evaluated — {status['note']}")
                if line != last_line:
                    print(line, flush=True)
                    last_line = line
            if status["done"]:
                break

        for task in pending:  # campaign over; stop waiting on stragglers
            task.cancel()
        dash.finish()
        event_log.close()
        print_report(await curator.report(), args.outdir)
        campaign_done = True
    except Exception as exc:  # noqa: BLE001
        # Manager close (the `async with` exit) re-raises the stored startup
        # exception of every failed agent — including members we already
        # reported as DROPPED at the readiness barrier — which would make a
        # successful campaign end in a traceback. Reduce those to one line;
        # anything before the report is a real failure and propagates.
        if not campaign_done:
            raise
        print(f"(manager close re-raised dropped-member errors: {exc})")


if __name__ == "__main__":
    asyncio.run(main())
