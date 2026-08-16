# Running the Li-ion committee campaign on Polaris (ALCF)

End-to-end instructions for the battery-materials variant of the
Academy × Rootstock query-by-committee demo: ~28 Materials Project seeds
(cathodes, anodes, solid electrolytes), a committee of foundation MLIPs served
by Rootstock from the shared Almanac install on Eagle, an autonomous Curator
that chases disagreement on delithiation / Li-hop / Li–TM antisite
perturbations, a live terminal dashboard, and an interactive HTML report.

Nothing here has to be built by you: the MLIP environments already exist on
Polaris (Almanac lists 51 verified checkpoints there). Your side is a Python
3.11 + `uv` environment, one seeds file, and a PBS job.

> Status: the workflow is validated end-to-end in `--mock` mode; the Polaris
> driver mirrors the working Sophia driver (same ALCF proxy / PBS quirks).
> First real run: expect ~20 min of model warm-up before the campaign starts,
> and check "Troubleshooting" if a member fails to seat.

---

## 0. What you need

| Requirement | Notes |
|---|---|
| ALCF account with Polaris access | `ssh <user>@polaris.alcf.anl.gov` (MobilePASS+ token) |
| A project allocation | the drivers use `-A Garden-AI`; change to your project |
| Materials Project API key | free, https://materialsproject.org/api — only needed **once**, to fetch seeds; not on the compute node |
| Hugging Face token (optional) | only if you want `uma-*` in the pool; put it in `~/.cache/huggingface/token` on Polaris |
| `uv` on Polaris | see step 1 |

Files from this repo that go to Polaris (everything is a self-contained
PEP 723 script — `uv run` resolves dependencies on first use):

```
academy_mlip_committee.py       # the campaign (agents, Curator, TUI)
report_html.py                  # interactive HTML report
visualize_committee.py          # static summary figure (PNG/SVG)
academy_committee_polaris.pbs   # PBS driver
battery_seeds.extxyz            # from fetch_battery_seeds.py (step 2)
```

---

## 1. One-time setup on Polaris

```bash
ssh <user>@polaris.alcf.anl.gov

# uv (user-local; ALCF login nodes need the proxy for outbound HTTP)
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128 HTTP_PROXY=http://proxy.alcf.anl.gov:3128
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'umask 002' >> ~/.bashrc          # shared-install friendliness (Rootstock docs)
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc   # home + Eagle are different filesystems
source ~/.bashrc
uv --version

# Working directory on Eagle (fast, project-shared); home works too.
mkdir -p /eagle/<project>/$USER/committee && cd $_
```

Confirm Rootstock can see the Polaris install and its verified checkpoints
(this needs no GPU — run it on the login node):

```bash
uv run --with 'rootstock>=1.2' rootstock resolve --cluster polaris --json | head -40
```

You should see the install root on Eagle and a checkpoint list. The
campaign builds its committee from `COMMITTEE_PREFERENCE` in
`academy_mlip_committee.py`, keeping only checkpoints the manifest marks
**verified on Polaris** and at most one per environment. As of writing that
yields (in order): `mace-mp-0-medium`, `tensornet-matpes-pbe-2025-2`,
`orb-v3-conservative-inf-omat`, `chgnet-default`, `grace-2l-smax-omat-large`,
`pet-omatpes-l` — six distinct training lineages (MPtrj ×2, MatPES, OMat ×2,
OMatPES). SevenNet and MatterSim are not verified on Polaris and are skipped
automatically. Override with `--committee id1,id2,...` if you want a
particular set (e.g. add `uma-s-1p1`, gated: needs the HF token).

Optional smoke test of one model on a compute node (interactive, 15 min):

```bash
qsub -I -A <project> -q debug -l select=1:system=polaris -l walltime=00:15:00 -l filesystems=home:eagle
# on the node:
export PATH=$HOME/.local/bin:$PATH HTTPS_PROXY=http://proxy.alcf.anl.gov:3128 HTTP_PROXY=$HTTPS_PROXY
uv run --with 'rootstock>=1.2' --with ase python - <<'EOF'
from ase.build import bulk
from rootstock import RootstockCalculator
atoms = bulk("Cu", "fcc", a=3.6) * (3, 3, 3)
with RootstockCalculator(cluster="polaris", checkpoint="mace-mp-0-medium", device="cuda") as calc:
    atoms.calc = calc
    print(atoms.get_potential_energy())
EOF
```

The first call pays worker spawn + model load (minutes on a cold Lustre
cache); a second call is fast. That is exactly why the campaign uses
one long-lived agent per model.

---

## 2. Fetch the seed structures (anywhere with internet + MP key)

On your laptop (or a Polaris login node with the proxy exported):

```bash
export MP_API_KEY=...
uv run fetch_battery_seeds.py            # writes battery_seeds.extxyz
```

It resolves ~28 named compounds to their lowest-hull, experimentally observed
MP entries — layered LiCoO₂/LiNiO₂/LiMnO₂/Li₂MnO₃, spinels LiMn₂O₄ and
LiNi₀.₅Mn₁.₅O₄, olivines LiFePO₄/LiMnPO₄/LiCoPO₄, tavorites, Li₂FeSiO₄,
Li₃V₂(PO₄)₃, LiTiS₂, LiVO₂; anodes Li, LiC₆, Li₁₅Si₄, LTO; electrolytes LLZO,
LGPS, Li₃PS₄, Li₆PS₅Cl, LiTi₂(PO₄)₃, Li₃OCl, Li₃N, Li₂S — with `mp_id`,
`e_above_hull`, space group and a `role` tag riding along into every report.
Anything MP can't resolve within `--ehull-max 0.08` is listed as missing at
the end; the campaign runs fine with what resolved.

Copy it (and the scripts) to Polaris:

```bash
scp academy_mlip_committee.py report_html.py visualize_committee.py \
    academy_committee_polaris.pbs battery_seeds.extxyz \
    <user>@polaris.alcf.anl.gov:/eagle/<project>/<user>/committee/
```

---

## 3. Dry run without a GPU (30 s, on the login node)

Exercises the full agent topology — staggered warm-ups, quorum start,
mid-campaign joins, the battery perturbation arms, the event stream, report
and HTML — with pair-potential stand-ins:

```bash
uv run academy_mlip_committee.py --mock --builtin battery --rounds 6 --round-pause 3 \
    --pool-size 5 --quorum 3 --outdir mock_battery
uv run report_html.py mock_battery/committee_report.json      # open mock_battery/committee_report.html
```

In a terminal you get the live dashboard; add `--no-tui` for plain lines.
The mock disagreement pattern is synthetic — the point is that everything
downstream works before you spend GPU time.

---

## 4. The real run

### 4a. Batch (recommended first run)

Edit `#PBS -A` in `academy_committee_polaris.pbs` to your project, then:

```bash
qsub -v EXTRA_ARGS="--seeds battery_seeds.extxyz --rounds 8 --batch-size 16" \
     academy_committee_polaris.pbs
```

What the driver does: exports the ALCF proxy (compute nodes have no direct
outbound access), points `ROOTSTOCK_WORKER_LOG` at a real file so worker
stdout/stderr are watchable, forwards an HF token if present, runs the
campaign with `--cluster polaris --timeout 2700 --round-pause 60`, then
generates `committee_report.html` and `committee_summary.png` in the
results directory. `debug` queue gives you 1 h on 1 node, which is enough
for 6–8 rounds; for more use `-q prod -l walltime=02:00:00`.

Watch it:

```bash
qstat -u $USER
tail -f academy-committee.o<jobid>            # plain progress lines
tail -f rootstock-workers-<jobid>.log         # per-model worker logs (model load, CUDA init…)
tail -f committee_results_<jobid>/committee_events.jsonl   # the Curator's event stream
```

Timeline to expect on a cold node: `launching 6 member agents…` → members
report `warm in NNNs` one by one over 5–25 min → `quorum of 3 reached` →
rounds begin (each round is seconds; `--round-pause 60` stretches the
campaign so stragglers still have rounds to join) → late members `joined the
committee (round k)` → report.

### 4b. Interactive, with the live dashboard

```bash
qsub -I -A <project> -q debug -l select=1:system=polaris -l walltime=01:00:00 -l filesystems=home:eagle
# on the node:
cd /eagle/<project>/$USER/committee
export PATH=$HOME/.local/bin:$PATH
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128 HTTP_PROXY=$HTTPS_PROXY http_proxy=$HTTPS_PROXY https_proxy=$HTTPS_PROXY
export ROOTSTOCK_WORKER_LOG=$PWD/rootstock-workers-interactive.log
uv run academy_mlip_committee.py --cluster polaris --seeds battery_seeds.extxyz \
    --rounds 8 --batch-size 16 --round-pause 60 --timeout 2700 --outdir committee_results_live
```

The dashboard shows each member's state (warming → seated → scoring, strikes,
dropped), warm-up time, evals and last-batch latency; the σ(F) trend; the
arms being escalated; and the event log. On exit it writes
`committee_dashboard.svg` — the final board as an image. Record the session
with `asciinema rec` (or `script`) if you want the GIF.

### GPU notes

- All members share **GPU 0** by default (`--device cuda`). The models are
  small and the cells are 32–120 atoms; contention is negligible next to
  model load. If you want to spread members over the node's 4 A100s, run
  four separate campaigns (e.g. one per seed subset) with
  `CUDA_VISIBLE_DEVICES=0..3` — the script itself is single-device.
- Warm-up dominates cost: ~5–25 min per model on a cold Lustre cache, in
  parallel. `--timeout 2700` covers the slowest; if a member misses it, it is
  dropped and the campaign continues with the rest.
- Rough budget for this campaign: **≤ 1 node-hour** (all warm-ups + 8 rounds
  × 16 structures × 6 models is a few minutes of actual inference).

---

## 5. Outputs

```
committee_results_<jobid>/
├── committee_report.json         # everything: committee, seeds+provenance, history, arms, selected (with per-member forces)
├── selected_structures.extxyz    # top-k structures to label with DFT, provenance in info
├── committee_events.jsonl        # Curator event stream (joins, strikes, per-round best/escalation)
├── committee_report.html         # interactive report (3D viewer, force arrows per model, outlier table)
├── committee_summary.png/.svg    # static figure
└── committee_dashboard.svg       # final TUI board (interactive runs)
```

`committee_report.html` is a single self-contained file — `scp` it home and
open it. What to look at first: the arm ranking (which materials/perturbation
families the committee can't agree on — expect delithiated and Li-hop
configurations of the polyanion cathodes and the sulfide electrolytes), and
for the top structures the "who's the outlier" table: whether the odd model
out is consistently the same lineage.

Re-generate any output offline:

```bash
uv run report_html.py committee_results_<jobid>/committee_report.json --title "Li-ion committee, Polaris"
uv run visualize_committee.py committee_results_<jobid>
```

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `uv: command not found` in the job | `~/.local/bin` not on PATH under PBS — the driver exports it; for interactive sessions do it yourself |
| `uv run` hangs resolving deps on the node | proxy not exported (compute nodes have no outbound network) — the driver sets `HTTP(S)_PROXY`; interactive: export them |
| a member is `DROPPED — startup failed` | read `rootstock-workers-<jobid>.log`; typical: gated checkpoint without HF token (`uma-*`), or a checkpoint not verified on Polaris (use `rootstock resolve --cluster polaris`) — the campaign continues without it |
| everything warms slowly / timeouts | cold Lustre; raise `--timeout`, or first run `rootstock` on one model interactively to warm the cache; keep `--round-pause` so late joiners still get rounds |
| `fewer than 2 committee members started` | quorum can't form — usually the proxy/token issue above hitting every member; check the worker log |
| TUI garbled in the `.o` file | expected — the driver disables it (`stdout` isn't a TTY); use `qsub -I` for the dashboard |
| Materials Project fetch fails | `MP_API_KEY` unset, or run on a compute node (no network) — fetch on laptop/login node |
| Permissions errors under the shared install | make sure `umask 002` and `UV_LINK_MODE=copy` are in `~/.bashrc` (Rootstock cluster-setup notes) |

---

## 7. What to change for a different campaign

- **Seeds**: any extxyz with `seed_name`/`mp_id` in `info` (`fetch_mp_structures.py --chemsys Cu-Au` for the original alloy demo). Seeds containing Li/Na/K/Mg automatically get the `delith` / `hop` / `antisite` arms; others get `vacancy` / `swap`.
- **Committee**: `--committee` list, or edit `COMMITTEE_PREFERENCE`; `--pool-size` / `--quorum` control how many are launched and how many must be warm to start.
- **Campaign length**: `--rounds`, `--batch-size`, `--top-k`; `--round-pause` only matters for letting slow members join.
- **Cluster**: `--cluster delta|sophia|perlmutter|polaris` — same script; the drivers only differ in scheduler boilerplate.
