# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "academy-py>=1.0,<2",
#     "groundhog-hpc>=0.9.3",
#     "rootstock>=1.6.4",
#     "ase>=3.22",
#     "numpy",
# ]
#
# [tool.uv]
# exclude-newer = "2026-09-12T00:00:00Z"
#
# # One `[tool.hog.*]` table per facility/Globus Compute MEP.
# # Keys other than `endpoint` are passed as endpoint user configuration.
# # NOTE: you will need to replace the `account` fields with your own allocation(s)
# # in order to run this script.
#
# [tool.hog.delta]
# endpoint = "4a266c83-3c68-4a75-99b3-9c7459f2f7ef"
# account = "bhhl-delta-gpu"
# partition = "gpuA100x4"
# walltime = "06:00:00"
# cores_per_node = 16
# mem_per_node = 64
# scheduler_options = "#SBATCH --gpus-per-node=1"
# # Supplying any worker_init replaces this endpoint's default worker setup,
# # which builds a venv holding the Globus Compute worker; groundhog always
# # supplies a worker_init so rebuild that venv here the way the endpoint itself
# # does.
# worker_init = """
# export PATH="$HOME/.local/bin:$PATH"
# command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# uv venv --allow-existing "$HOME/.globus_compute/.venvs/py3.13-gce4.16.0" --python 3.13
# source "$HOME/.globus_compute/.venvs/py3.13-gce4.16.0/bin/activate"
# uv pip install --quiet globus-compute-endpoint==4.16.0
# """
#
# [tool.hog.polaris]
# endpoint = "9a947ba5-f537-4681-acf3-cc66485aadec"
# account = "Rootstock"
# queue = "capacity"
# walltime = "06:00:00"
# scheduler_options = "#PBS -l filesystems=home:eagle"
# available_accelerators = 4
# worker_init = """\
#     export PATH=/opt/globus-compute-agent/venv-py313/bin:$PATH; \
#     export CC=gcc; \
#     export http_proxy=http://proxy.alcf.anl.gov:3128; \
#     export https_proxy=http://proxy.alcf.anl.gov:3128; \
#     export HTTP_PROXY=http://proxy.alcf.anl.gov:3128; \
#     export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128;"""
# ///
"""A committee of machine-learning interatomic potentials (MLIPs), distributed
across HPC facilities, iteratively searching for structures that models disagree
about.

Each committee member is an Academy agent hosting one MLIP through Rootstock,
kept warm on a GPU for the whole campaign. Members run on NCSA Delta and ALCF
Polaris as Globus Compute tasks (joining the committee as they come online);
groundhog ships this script to each site and builds its environment there from
the header above.

A Curator agent running locally drives an adaptive query-by-committee loop: it
perturbs Cu-Au crystals, asks every member for energies and forces, scores each
structure by how much the members disagree, and steers the next round toward the
perturbations that produced the most disagreement. The output is the ranked set
of structures most worth investigating with DFT.

Run it:

    hog run federated_mlip_committee.py probe -- --sites delta,polaris
    hog run federated_mlip_committee.py -- --rounds 4

The first command inspects each facility from inside a Globus Compute task
(GPU, egress, which checkpoints Rootstock has verified there) and warms the
environment cache. The second runs the campaign. 

Member start-up is dominated by Rootstock loading a model from a cold
filesystem and ranges from two minutes to over an hour depending on the
cluster/shared filesystem demand. The campaign therefore starts as soon as
--quorum members are ready and seats the rest as they come up; --round-pause
stretches the rounds so late members still get to vote. --startup-timeout bounds
each member's start, and each member's worker log lands in
~/.federated_committee/<checkpoint>.log on its cluster (its tail is reported
back if the member fails to start). Members are only launched where Rootstock's
manifest marks the checkpoint verified (--allow-unverified overrides). TASK lines
report each Globus Compute task's id and scheduler-side status as it changes,
and a "waiting on" line every five minutes lists what is still pending.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import subprocess
import sys
import time
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from pathlib import Path

import groundhog_hpc as hog
from groundhog_hpc.compute import get_task_status
import numpy as np
from academy.agent import Agent, action, loop
from academy.exchange import HttpExchangeFactory
from academy.exchange.cloud.client import DEFAULT_EXCHANGE_URL
from academy.handle import Handle
from academy.logging.recommended import recommended_logging
from academy.manager import Manager
from ase import Atoms
from ase.build import bulk
from ase.io import write

# Two members on Delta (slow cold filesystem: the two smallest), four on
# Polaris; one model family per member so disagreement is never the same
# model twice. Every checkpoint loads with its verified default arguments.
DEFAULT_COMMITTEE = (
    "mace-mp-0-medium@delta,sevennet-omat@delta,"
    "orb-v3-conservative-inf-omat@polaris,grace-2l-smax-omat-large@polaris,"
    "tensornet-matpes-pbe-2025-2@polaris,pet-omatpes-l@polaris"
)


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def line_buffered_stdout() -> None:
    """Flush every line. Piped through `tee`, stdout is otherwise block-buffered
    and a driver that waits hours for a queue prints nothing until it exits."""
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)

# --------------------------------------------------------------------------
# Structures: ASE Atoms <-> plain-dicts
# --------------------------------------------------------------------------


def atoms_to_payload(atoms: Atoms) -> dict:
    return {
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
        "info": dict(atoms.info),
    }


def payload_to_atoms(payload: dict) -> Atoms:
    return Atoms(
        numbers=payload["numbers"],
        positions=payload["positions"],
        cell=payload["cell"],
        pbc=payload["pbc"],
        info=payload.get("info", {}),
    )


def build_seeds() -> dict[str, Atoms]:
    """Cu-Au phases as 2x2x2 supercells of the conventional fcc cell (32 atoms).

    Elements plus the ordered intermetallics: models trained on the same data
    tend to agree on these, so disagreement has to be earned by perturbation.
    """
    cu = bulk("Cu", "fcc", a=3.615, cubic=True)
    au = bulk("Au", "fcc", a=4.078, cubic=True)

    cu3au = bulk("Cu", "fcc", a=3.75, cubic=True)  # L1_2: Au on the corner site
    cu3au.symbols[0] = "Au"

    cuau = bulk("Cu", "fcc", a=3.96, cubic=True)  # L1_0: alternating (001) layers
    for i, z in enumerate(cuau.positions[:, 2]):
        if abs(z) < 1e-6:
            cuau.symbols[i] = "Au"
    cuau.set_cell(cuau.cell.array * [1.0, 1.0, 0.93], scale_atoms=True)

    return {
        "Cu": cu.repeat(2),
        "Au": au.repeat(2),
        "Cu3Au": cu3au.repeat(2),
        "CuAu": cuau.repeat(2),
    }


# Largest magnitude the Curator may escalate each arm to (same units as Arm.magnitude).
ARM_CAPS = {"rattle": 0.35, "strain": 0.08, "swap": 0.5, "vacancy": 0.1}


@dataclass
class Arm:
    """One family of perturbations and how hard the Curator is pulling it."""

    magnitude: float
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


def perturb(seed: Atoms, arm: str, magnitude: float, rng: np.random.Generator) -> Atoms:
    atoms = seed.copy()
    if arm == "rattle":
        atoms.positions += rng.normal(0.0, magnitude, atoms.positions.shape)
    elif arm == "strain":
        strain = 1.0 + rng.uniform(-magnitude, magnitude, size=3)
        atoms.set_cell(atoms.cell.array * strain, scale_atoms=True)
    elif arm == "swap":
        # Composition-preserving antisite swaps: the order -> disorder story.
        symbols = np.array(atoms.get_chemical_symbols())
        species = sorted(set(symbols))
        if len(species) >= 2:
            a = np.flatnonzero(symbols == species[0])
            b = np.flatnonzero(symbols == species[1])
            n = max(1, round(magnitude * min(len(a), len(b))))
            for i, j in zip(rng.permutation(a)[:n], rng.permutation(b)[:n]):
                symbols[i], symbols[j] = symbols[j], symbols[i]
            atoms.set_chemical_symbols(symbols.tolist())
        atoms.positions += rng.normal(0.0, 0.02, atoms.positions.shape)
    elif arm == "vacancy":
        del atoms[int(rng.integers(len(atoms)))]
        atoms.positions += rng.normal(0.0, magnitude, atoms.positions.shape)
    return atoms


def disagreement(results: dict[str, dict]) -> tuple[float, float]:
    """Committee disagreement on one structure.

    Returns the largest per-atom standard deviation of the force vector across
    members (eV/Angstrom, the usual query-by-committee uncertainty) and the
    spread of energy per atom (eV/atom).
    """
    forces = np.stack([np.asarray(r["forces"]) for r in results.values()])
    per_atom = np.linalg.norm(forces.std(axis=0), axis=1)
    energies = np.array([r["energy"] for r in results.values()]) / forces.shape[1]
    return float(per_atom.max()), float(energies.max() - energies.min())


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return ", ".join(sorted(set(out.split("\n")))).strip(", ") or "none"
    except Exception:
        return "none"


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------


class MLIPMember(Agent):
    """One MLIP, kept warm on one cluster for the whole campaign.

    Rootstock runs the model in its own pre-built environment as a worker
    process next to this agent; the agent only speaks ASE. Worker spawn and
    model load are paid once, at startup.
    """

    def __init__(
        self, checkpoint: str, site: str, device: str = "cuda", timeout: float = 1800.0
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.site = site
        self.device = device
        self.timeout = timeout
        self._calc = None
        self._warmup_s: float | None = None

    def _start(self):
        import os

        from rootstock import RootstockCalculator
        from rootstock.clusters import get_cluster
        from rootstock.environment import resolve_checkpoint, verify_kwargs_for

        # Rootstock's worker only writes its startup progress (page-cache
        # prewarm, model load) to the file named here; without it a slow
        # start is indistinguishable from a hung one.
        log = Path.home() / ".federated_committee" / f"{self.checkpoint}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("ROOTSTOCK_WORKER_LOG", str(log))
        try:
            # Load the model exactly the way the install verified it: a
            # multitask checkpoint, for one, needs the head its env source
            # declares for verification and has no default otherwise.
            root = get_cluster(self.site).root
            env_name = resolve_checkpoint(root, self.checkpoint, self.site).env_name
            setup_kwargs = verify_kwargs_for(root, env_name, self.checkpoint)
            calc = RootstockCalculator(
                checkpoint=self.checkpoint,
                cluster=self.site,
                device=self.device,
                timeout=self.timeout,
                setup_kwargs=setup_kwargs,
            )
            probe = bulk("Cu", "fcc", a=3.615)
            probe.calc = calc
            probe.get_forces()
        except Exception as exc:
            tail = log.read_text().splitlines()[-30:] if log.exists() else []
            raise RuntimeError(
                f"{exc}\n--- worker log {log} on {socket.gethostname()} ---\n"
                + ("\n".join(tail) or "(empty)")
            ) from exc
        return calc

    async def agent_on_startup(self) -> None:
        started = time.monotonic()
        self._calc = await self.agent_run_sync(self._start)
        self._warmup_s = time.monotonic() - started

    async def agent_on_shutdown(self) -> None:
        if self._calc is not None:
            self._calc.close()

    @action
    async def info(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "site": self.site,
            "host": socket.gethostname(),
            "gpu": gpu_name(),
            "warmup_s": self._warmup_s,
        }

    def _evaluate(self, payloads: list[dict]) -> list[dict]:
        results = []
        for payload in payloads:
            atoms = payload_to_atoms(payload)
            atoms.calc = self._calc
            results.append(
                {
                    "energy": float(atoms.get_potential_energy()),
                    "forces": atoms.get_forces().tolist(),
                }
            )
        return results

    @action
    async def evaluate(self, payloads: list[dict]) -> list[dict]:
        return await self.agent_run_sync(self._evaluate, payloads)


class Curator(Agent):
    """Runs the query-by-committee campaign over the member agents."""

    def __init__(
        self,
        members: dict[str, Handle],
        rounds: int,
        batch_size: int,
        top_k: int,
        seed: int,
        outdir: Path,
        round_pause: float = 0.0,
    ):
        super().__init__()
        self.members = dict(members)
        self.joined: dict[str, int] = {name: 0 for name in members}  # 0 = founding
        self.rounds = rounds
        self.round_pause = round_pause
        self.batch_size = batch_size
        self.top_k = top_k
        self.outdir = Path(outdir)
        self.rng = np.random.default_rng(seed)
        self.seeds = build_seeds()
        self.arms = {
            "rattle": Arm(0.05),  # Angstrom
            "strain": Arm(0.03),  # fractional
            "swap": Arm(0.125),  # fraction of the minority species swapped
            "vacancy": Arm(0.02),  # Angstrom of rattle around the vacancy
        }
        self.scored: list[tuple[float, Atoms, dict]] = []
        self.failures: dict[str, int] = {name: 0 for name in members}
        self.dropped: list[str] = []
        self.rounds_done = 0
        self.done = asyncio.Event()
        self.aborted: str | None = None

    def propose(self, round_no: int) -> list[Atoms]:
        weights = np.array([0.05 + arm.mean for arm in self.arms.values()])
        arm_names = list(self.arms)
        batch = []
        for _ in range(self.batch_size):
            arm = str(self.rng.choice(arm_names, p=weights / weights.sum()))
            seed_name = str(self.rng.choice(list(self.seeds)))
            atoms = perturb(
                self.seeds[seed_name], arm, self.arms[arm].magnitude, self.rng
            )
            atoms.info = {
                "seed": seed_name,
                "arm": arm,
                "magnitude": self.arms[arm].magnitude,
                "round": round_no,
            }
            batch.append(atoms)
        return batch

    def adapt(self, round_records: list[dict]) -> None:
        for record in round_records:
            self.arms[record["arm"]].scores.append(record["force_std_max"])
        leader = max(self.arms, key=lambda name: self.arms[name].mean)
        # Escalate the leading arm, but never past a physical magnitude:
        # uncapped, rattle runs away (disagreement grows with sigma, so the
        # leader stays the leader) and the committee ends up scoring
        # overlapping atoms.
        arm = self.arms[leader]
        arm.magnitude = min(arm.magnitude * 1.3, ARM_CAPS[leader])

    @action
    async def add_member(self, name: str, handle: Handle) -> int:
        """Seat a member that finished starting up mid-campaign; returns its first round."""
        self.members[name] = handle
        self.failures[name] = 0
        self.joined[name] = self.rounds_done + 1
        return self.joined[name]

    @loop
    async def campaign(self, shutdown: asyncio.Event) -> None:
        for round_no in range(1, self.rounds + 1):
            if shutdown.is_set():
                break
            if round_no > 1 and self.round_pause:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(shutdown.wait(), self.round_pause)
            batch = self.propose(round_no)
            payloads = [atoms_to_payload(atoms) for atoms in batch]
            names = list(self.members)
            replies = await asyncio.gather(
                *(self.members[name].evaluate(payloads) for name in names),
                return_exceptions=True,
            )

            results: dict[str, list[dict]] = {}
            for name, reply in zip(names, replies):
                if isinstance(reply, BaseException):
                    self.failures[name] += 1
                    print(
                        f"  {name}: round {round_no} failed ({type(reply).__name__}: {reply})"
                    )
                    if self.failures[name] >= 2:
                        print(f"  {name}: dropped from the committee")
                        self.dropped.append(name)
                        del self.members[name]
                else:
                    self.failures[name] = 0
                    results[name] = reply
            if len(results) < 2:
                self.aborted = "fewer than two members answered"
                break

            records = []
            for i, atoms in enumerate(batch):
                force_std_max, energy_spread = disagreement(
                    {name: reply[i] for name, reply in results.items()}
                )
                record = {
                    "id": f"r{round_no}-{i:02d}",
                    **atoms.info,
                    "natoms": len(atoms),
                    "force_std_max": force_std_max,
                    "energy_spread": energy_spread,
                    "members": sorted(results),
                    "energies": {name: reply[i]["energy"] for name, reply in results.items()},
                }
                atoms.info.update(
                    force_std_max=force_std_max, energy_spread=energy_spread
                )
                records.append(record)
                self.scored.append((force_std_max, atoms, record))
            self.adapt(records)
            self.rounds_done = round_no

            best = max(records, key=lambda r: r["force_std_max"])
            arms = ", ".join(f"{n} {a.mean:.3f}" for n, a in self.arms.items())
            print(
                f"{stamp()} round {round_no}/{self.rounds}: {len(batch)} structures x "
                f"{len(results)} members; most disagreement {best['force_std_max']:.3f} "
                f"eV/A on {best['seed']}/{best['arm']}; arms {arms}"
            )

        self.write_outputs()
        self.done.set()

    def ranking(self) -> list[dict]:
        ordered = sorted(self.scored, key=lambda item: item[0], reverse=True)
        return [record for _, _, record in ordered[: self.top_k]]

    def write_outputs(self) -> None:
        self.outdir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.scored, key=lambda item: item[0], reverse=True)
        top = [atoms for _, atoms, _ in ordered[: self.top_k]]
        write(self.outdir / "selected_structures.extxyz", top)
        (self.outdir / "committee_report.json").write_text(
            json.dumps(self.report_dict(), indent=2)
        )

    def report_dict(self) -> dict:
        return {
            "members": sorted(self.members),
            "joined_round": dict(sorted(self.joined.items())),
            "dropped": self.dropped,
            "rounds_done": self.rounds_done,
            "aborted": self.aborted,
            "structures_scored": len(self.scored),
            "arms": {
                name: {"magnitude": arm.magnitude, "mean_force_std": arm.mean}
                for name, arm in self.arms.items()
            },
            "top": self.ranking(),
            "history": [record for _, _, record in self.scored],
            "outdir": str(self.outdir),
        }

    @action
    async def status(self) -> dict:
        leader = max(self.scored, key=lambda item: item[0], default=None)
        return {
            "round": self.rounds_done,
            "rounds": self.rounds,
            "scored": len(self.scored),
            "leader": None
            if leader is None
            else {k: leader[2][k] for k in ("seed", "arm", "force_std_max")},
            "done": self.done.is_set(),
        }

    @action
    async def report(self) -> dict:
        return self.report_dict()


# --------------------------------------------------------------------------
# Groundhog: the tasks that run on the clusters, and the bridge to Academy
# --------------------------------------------------------------------------


@hog.function()
def run_task(fn, *args):
    """One Globus Compute task hosts one Academy agent for its whole lifetime."""
    return fn(*args)


def task_failure_text(exc: BaseException) -> str:
    """The error plus the tail of the task's stderr, where the real cause lives."""
    tail = "\n".join(str(getattr(exc, "stderr", "") or "").splitlines()[-20:])
    return f"{type(exc).__name__}: {exc}" + (f"\n{tail}" if tail else "")


def _print_task_failure(future: Future, site: str) -> None:
    if future.cancelled():
        return
    exc = future.exception()
    if exc is not None:
        print(f"--- task on {site} failed ---\n{task_failure_text(exc)}\n---")


class TaskWatch:
    """Reports each Globus Compute task's id and scheduler-side status.

    A task can sit for hours before a facility's user endpoint picks it up
    (`waiting-for-ep`) or its job starts (`waiting-for-nodes`); without this
    the driver is silent for exactly that long. Prints a line on every status
    change and a periodic summary of what is still pending; stops following a
    task once it is running.
    """

    SETTLED = {"running", "success", "failed"}

    def __init__(self, heartbeat_s: float = 300.0):
        self.heartbeat_s = heartbeat_s
        self._pending: dict[str, Future] = {}
        self._status: dict[str, str] = {}
        self._since: dict[str, float] = {}

    def add(self, label: str, future: Future) -> None:
        self._pending[label] = future
        self._since[label] = time.monotonic()

    @staticmethod
    def _lookup(future) -> str:
        task_id = getattr(future, "task_id", None)
        if task_id is None:
            return "submitting"
        try:
            return get_task_status(task_id).get("status", "unknown")
        except Exception as exc:
            return f"status lookup failed ({type(exc).__name__})"

    async def run(self, poll_s: float = 20.0) -> None:
        last_heartbeat = time.monotonic()
        while True:
            for label, future in list(self._pending.items()):
                if future.done():
                    self._pending.pop(label)
                    continue
                status = await asyncio.to_thread(self._lookup, future)
                if status != self._status.get(label):
                    self._status[label] = status
                    task_id = getattr(future, "task_id", None) or "id pending"
                    print(f"{stamp()} TASK {label}: {status} (task {task_id})")
                if status in self.SETTLED:
                    self._pending.pop(label)
            now = time.monotonic()
            if self._pending and now - last_heartbeat >= self.heartbeat_s:
                last_heartbeat = now
                waits = ", ".join(
                    f"{label} {self._status.get(label, '?')} "
                    f"for {(now - self._since[label]) / 60:.0f} min"
                    for label in self._pending
                )
                print(f"{stamp()} waiting on: {waits}")
            await asyncio.sleep(poll_s)


class GroundhogExecutor(Executor):
    """An Academy executor whose workers are groundhog tasks on one facility.

    Academy's Manager hands every executor the function that runs an agent
    and the agent's launch spec; this one forwards them to run_task on the
    site's Globus Compute endpoint, so the agent's environment is built there
    from this script's header.
    """

    def __init__(self, site: str, watch: TaskWatch | None = None):
        self.site = site
        self.watch = watch
        # The Manager's submit call carries no agent name; the launcher sets
        # this just before `manager.launch` so the task is reported by member.
        self.next_label: str | None = None

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future = run_task.submit(fn, *args, endpoint=self.site, **kwargs)
        future.add_done_callback(lambda done: _print_task_failure(done, self.site))
        if self.watch is not None:
            self.watch.add(self.next_label or f"agent on {self.site}", future)
        self.next_label = None
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        pass


@hog.function()
def probe_site(site: str) -> dict:
    """What does this facility look like from inside a Globus Compute task?"""
    import importlib.metadata
    import os
    import platform
    import shutil
    import urllib.error
    import urllib.request
    from pathlib import Path

    from rootstock.clusters import get_cluster
    from rootstock.manifest import is_verified, load_manifest

    root = get_cluster(site).root
    manifest = load_manifest(Path(root))
    verified = sorted(
        ckpt
        for env in (manifest.environments.values() if manifest else [])
        for ckpt, record in env.checkpoints.items()
        if is_verified(env, record, site)
    )
    try:
        urllib.request.urlopen(DEFAULT_EXCHANGE_URL, timeout=20)
        exchange = "reachable"
    except urllib.error.HTTPError:
        exchange = "reachable"  # any HTTP status means we got through
    except Exception as exc:
        exchange = f"unreachable ({type(exc).__name__})"
    return {
        "host": socket.gethostname(),
        "gpu": gpu_name(),
        "uv": shutil.which("uv") or "not on PATH",
        "https_proxy": os.environ.get("https_proxy", ""),
        "exchange": exchange,
        "python": platform.python_version(),
        "rootstock": importlib.metadata.version("rootstock"),
        "root": str(root),
        "verified": verified,
    }


# --------------------------------------------------------------------------
# Harnesses (these run on your machine)
# --------------------------------------------------------------------------


def parse_committee(spec: str) -> dict[str, tuple[str, str]]:
    committee = {}
    for entry in filter(None, (s.strip() for s in spec.split(","))):
        checkpoint, _, site = entry.partition("@")
        if not site:
            raise ValueError(f"member {entry!r} must look like checkpoint@site")
        committee[entry] = (checkpoint, site)
    return committee


def print_probe(site: str, facts: dict) -> None:
    print(
        f"{site}: {facts['host']}  gpu={facts['gpu']}  uv={facts['uv']}  "
        f"exchange={facts['exchange']}"
    )
    print(
        f"    python {facts['python']}, rootstock {facts['rootstock']} at {facts['root']}; "
        f"{len(facts['verified'])} verified checkpoints"
    )


@hog.harness()
def probe(sites: str = "delta,polaris") -> None:
    """Inspect each facility from inside a task and warm its environment cache."""
    line_buffered_stdout()
    names = [s.strip() for s in sites.split(",") if s.strip()]
    futures = {site: probe_site.submit(site, endpoint=site) for site in names}
    for site, future in futures.items():
        try:
            print_probe(site, future.result())
        except Exception as exc:
            print(f"{site}: probe failed\n{task_failure_text(exc)}")


async def campaign(
    committee: dict[str, tuple[str, str]],
    rounds: int,
    batch_size: int,
    top_k: int,
    seed: int,
    outdir: Path,
    startup_timeout: float,
    device: str,
    quorum: int,
    round_pause: float,
    allow_unverified: bool,
) -> None:
    sites = sorted({site for _, site in committee.values()})
    watch = TaskWatch()
    watch_task = asyncio.create_task(watch.run())
    executors = {site: GroundhogExecutor(site, watch) for site in sites}
    report = None
    try:
        async with await Manager.from_exchange_factory(
            HttpExchangeFactory(),
            executors=executors,
            # A member whose task dies is relaunched once on the same mailbox,
            # so its handle stays valid. The long-poll to the exchange has been
            # seen to drop mid-stream from behind a facility web proxy. Only
            # once: a model that takes an hour to load should not be retried
            # into the campaign's walltime.
            max_retries=1,
            log_config=recommended_logging("WARNING"),
        ) as manager:
            handles: dict[str, Handle] = {}
            ready: dict[str, Handle] = {}
            waiting: set[asyncio.Task] = set()  # one readiness task per launched member

            async def await_ready(name: str, handle: Handle) -> tuple[str, Handle | None]:
                try:
                    # The member's own calculator timeout is startup_timeout; give
                    # its failure report time to arrive before giving up here.
                    info = await asyncio.wait_for(handle.info(), startup_timeout + 120)
                except Exception as exc:
                    print(f"{stamp()} DROPPED {name}: {type(exc).__name__}: {exc}")
                    return name, None
                print(
                    f"{stamp()} READY {name} on {info['host']} ({info['gpu']}) "
                    f"after {info['warmup_s']:.0f} s"
                )
                return name, handle

            async def bring_up(site: str) -> None:
                """Probe one facility, then launch its members the moment it answers.

                Sites come up independently: a slow queue at one facility must
                not delay the others, least of all a facility whose models take
                an hour to load.
                """
                members = {n: c for n, (c, s) in committee.items() if s == site}
                print(f"{stamp()} Probing {site} for {', '.join(members)}...")
                probe = probe_site.submit(site, endpoint=site)
                watch.add(f"probe {site}", probe)
                try:
                    facts = await asyncio.wrap_future(probe)
                except Exception as exc:
                    print(f"{stamp()} {site}: probe failed, skipping its members")
                    print(task_failure_text(exc))
                    return
                print_probe(site, facts)
                launched = 0
                for name, checkpoint in members.items():
                    if checkpoint not in facts["verified"]:
                        if not allow_unverified:
                            print(f"{stamp()} SKIP {name}: not verified on {site}")
                            continue
                        print(
                            f"{stamp()} WARNING {name}: not verified on {site}, "
                            "trying anyway (--allow-unverified)"
                        )
                    executors[site].next_label = name
                    handles[name] = await manager.launch(
                        MLIPMember,
                        args=(checkpoint, site, device, startup_timeout),
                        executor=site,
                        name=name,
                    )
                    waiting.add(asyncio.create_task(await_ready(name, handles[name])))
                    launched += 1
                if launched:
                    print(f"{stamp()} Launched {launched} of {len(members)} members on {site}")
                    return
                requested = {c for c, _ in committee.values()}
                alternatives = [c for c in facts["verified"] if c not in requested]
                print(
                    f"{stamp()} WARNING: {site} contributes no members; every requested "
                    f"checkpoint was skipped. Verified there now ({len(alternatives)} "
                    f"others): {', '.join(alternatives[:12])}"
                    f"{', ...' if len(alternatives) > 12 else ''}. "
                    "Rerun with --members ... or --allow-unverified."
                )

            bringing_up = {asyncio.create_task(bring_up(site)) for site in sites}

            async def next_ready() -> tuple[str, Handle] | None:
                """The next member to finish starting up; None once no more can."""
                while True:
                    pending_sites = {t for t in bringing_up if not t.done()}
                    if not waiting and not pending_sites:
                        return None
                    finished, _ = await asyncio.wait(
                        waiting | pending_sites, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in finished:
                        if task in waiting:
                            waiting.discard(task)
                            name, handle = task.result()
                            if handle is not None:
                                return name, handle
                        elif task.exception() is not None:
                            print(f"{stamp()} site bring-up failed: {task.exception()!r}")

            # Start the campaign once a quorum is warm; the rest join as they come up.
            while len(ready) < quorum:
                arrival = await next_ready()
                if arrival is None:
                    break
                ready[arrival[0]] = arrival[1]
            if len(ready) < 2:
                print("Fewer than two members came up; stopping.")
                return

            founders = ", ".join(sorted(ready))
            print(f"{stamp()} Campaign starts with {len(ready)} members: {founders}")
            curator = await manager.launch(
                Curator,
                args=(ready, rounds, batch_size, top_k, seed, outdir, round_pause),
                name="curator",
            )

            async def seat_stragglers() -> None:
                while (arrival := await next_ready()) is not None:
                    name, handle = arrival
                    joined = await curator.add_member(name, handle)
                    print(f"{stamp()} JOINED {name} from round {joined}")

            seating = asyncio.create_task(seat_stragglers())
            while not (await curator.status())["done"]:
                await asyncio.sleep(5)
            for task in {seating, *waiting, *bringing_up}:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await seating
            report = await curator.report()
            print_report(report)
            still_starting = [n for n in handles if n not in report["joined_round"]]
            if still_starting:
                # Their shutdown request is already queued, so they exit on their
                # own once loaded; results are on disk, so Ctrl-C here loses nothing.
                print(
                    f"{stamp()} Campaign over; waiting for members that never finished "
                    f"starting to shut down (Ctrl-C is safe): {', '.join(still_starting)}"
                )
    except Exception as exc:
        # Manager.close re-raises the startup error of any member that was
        # dropped; the failure itself was already printed above.
        print(f"(cleanup: {task_failure_text(exc).splitlines()[0]})")
    finally:
        watch_task.cancel()

    if report is None:
        raise SystemExit(1)


def print_report(report: dict) -> None:
    print()
    committee = ", ".join(
        name if r == 0 else f"{name} (joined round {r})"
        for name, r in report["joined_round"].items()
    )
    print(f"Committee: {committee}")
    if report["dropped"]:
        print(f"Dropped: {', '.join(report['dropped'])}")
    print(
        f"{report['structures_scored']} structures over {report['rounds_done']} rounds"
    )
    arms = ", ".join(
        f"{name} {arm['mean_force_std']:.3f} @ {arm['magnitude']:.3g}"
        for name, arm in report["arms"].items()
    )
    print(f"Arms (mean force std, final magnitude): {arms}")
    print(f"Top {len(report['top'])} structures to label:")
    for i, rec in enumerate(report["top"], 1):
        print(
            f"  {i:2d}. {rec['seed']:5s} {rec['arm']:8s} round {rec['round']}  "
            f"force std {rec['force_std_max']:.3f} eV/A  "
            f"energy spread {rec['energy_spread'] * 1000:.1f} meV/atom"
        )
    print(
        f"Written to {report['outdir']}/ (selected_structures.extxyz, committee_report.json)"
    )


@hog.harness()
def main(
    members: str = DEFAULT_COMMITTEE,
    rounds: int = 4,
    batch_size: int = 12,
    top_k: int = 8,
    seed: int = 7,
    outdir: Path = Path("committee_results"),
    startup_timeout: float = 1800.0,
    device: str = "cuda",
    quorum: int = 2,
    round_pause: float = 0.0,
    allow_unverified: bool = False,
) -> None:
    """Run the federated query-by-committee campaign."""
    line_buffered_stdout()
    asyncio.run(
        campaign(
            parse_committee(members),
            rounds,
            batch_size,
            top_k,
            seed,
            outdir,
            startup_timeout,
            device,
            quorum,
            round_pause,
            allow_unverified,
        )
    )
