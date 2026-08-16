# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Interactive, self-contained HTML report for a committee campaign.

Reads the ``committee_report.json`` written by academy_mlip_committee.py and
emits one HTML file with no external dependencies (own canvas 3D renderer,
inline SVG charts) — it opens from a laptop, a login node, or a static host.

    uv run report_html.py committee_results/committee_report.json
    uv run report_html.py committee_results/committee_report.json -o report.html

What you get:
  * the campaign scatter (round vs. committee force disagreement) — hover any
    point, click a selected one to load it;
  * "where does the committee disagree" arm ranking — click an arm to
    highlight its structures;
  * a 3D viewer for every selected structure: drag to rotate, scroll to zoom,
    color atoms by element or by per-atom force disagreement, and toggle each
    committee member's force arrows so you can *see* which model points the
    other way on which atom;
  * a per-structure "who's the outlier" panel: each member's RMS force
    deviation from the committee mean and its energy offset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Jmol CPK-ish element colors for the handful of elements likely to appear.
ELEMENT_COLORS = {
    1: "#ffffff", 3: "#cc80ff", 5: "#ffb5b5", 6: "#909090", 7: "#3050f8",
    8: "#ff0d0d", 9: "#90e050", 15: "#ff8000", 16: "#ffff30", 17: "#1ff01f",
    57: "#70d4ff", 11: "#ab5cf2", 12: "#8aff00", 13: "#bfa6a6", 14: "#f0c8a0", 19: "#8f40d4",
    20: "#3dff00", 22: "#bfc2c7", 23: "#a6a6ab", 24: "#8a99c7", 25: "#9c7ac7",
    26: "#e06633", 27: "#f090a0", 28: "#50d050", 29: "#c88033", 30: "#7d80b0",
    31: "#c28f8f", 32: "#668f8f", 40: "#94e0e0", 41: "#73c2c9", 42: "#54b5b5",
    44: "#248f8f", 45: "#0a7d8c", 46: "#006985", 47: "#c0c0c0", 48: "#ffd98f",
    49: "#a67573", 50: "#668080", 51: "#9e63b5", 55: "#57178f", 56: "#00c900",
    72: "#4dc2ff", 73: "#4da6ff", 74: "#2194d6", 75: "#267dab", 76: "#266696",
    77: "#175487", 78: "#d0d0e0", 79: "#ffd123", 80: "#b8b8d0", 82: "#575961",
    83: "#9e4fb5",
}
COVALENT_RADII = {  # Å, Cordero et al.
    1: 0.31, 3: 1.28, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07,
    16: 1.05, 17: 1.02, 57: 2.07, 11: 1.66, 12: 1.41, 13: 1.21,
    14: 1.11, 19: 2.03, 20: 1.76, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39,
    26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20,
    40: 1.75, 41: 1.64, 42: 1.54, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45,
    48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39, 55: 2.44, 56: 2.15, 72: 1.75,
    73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36,
    80: 1.32, 82: 1.46, 83: 1.48,
}
SYMBOLS = {
    1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
    17: "Cl", 57: "La", 11: "Na", 12: "Mg", 13: "Al",
    14: "Si", 19: "K", 20: "Ca", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge",
    40: "Zr", 41: "Nb", 42: "Mo", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag",
    48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 55: "Cs", 56: "Ba", 72: "Hf",
    73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au",
    80: "Hg", 82: "Pb", 83: "Bi",
}


def build_data(report: dict, title: str | None) -> dict:
    members = sorted(report["committee"], key=lambda n: (report["committee"][n], n))
    selected = []
    for cand in report["selected"]:
        p = cand["payload"]
        forces = cand.get("member_forces", {})
        # Per-atom disagreement exactly as the Curator scores it:
        # norm over xyz of the population std across members.
        per_atom = None
        if len(forces) >= 2:
            f = np.array([forces[m] for m in members if m in forces])
            per_atom = np.linalg.norm(f.std(axis=0), axis=1).round(4).tolist()
        selected.append({
            "id": cand["id"], "round": cand["round"], "seed": cand["seed"],
            "transform": cand["transform"], "magnitude": cand["magnitude"],
            "qbc": cand["qbc_force_std"], "dE": cand["energy_spread_per_atom"],
            "energies": cand["energies"],
            "numbers": p["numbers"], "positions": p["positions"],
            "cell": p["cell"], "info": p.get("info", {}),
            "member_forces": forces, "per_atom_sigma": per_atom,
        })
    elements = sorted({z for c in selected for z in c["numbers"]})
    return {
        "title": title or "Multi-MLIP committee report",
        "members": members,
        "joined": report["committee"],
        "never_joined": report.get("never_joined", []),
        "dropped": report.get("dropped_members", []),
        "rounds": report["rounds_completed"],
        "n_evaluated": report["n_evaluated"],
        "seeds": report["seeds"],
        "history": report["history"],
        "arms": report["arms"],
        "selected": selected,
        "elements": {z: {"symbol": SYMBOLS.get(z, str(z)),
                         "color": ELEMENT_COLORS.get(z, "#ff1493"),
                         "radius": COVALENT_RADII.get(z, 1.4)} for z in elements},
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#fcfcfb; --panel:#ffffff; --border:#e6e5e0; --grid:#efeeea;
  --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8984;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
  --sel:#0b0b0b;
  --seq100:#cde2fb; --seq700:#0d366b;
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.4}
body{padding:20px 24px 40px}
h1{font-size:22px;font-weight:600;margin:0 0 4px;letter-spacing:-0.01em}
h2{font-size:14px;font-weight:600;margin:0 0 8px;color:var(--ink)}
.sub{color:var(--ink2);font-size:13px;margin:0 0 2px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 0}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border:1px solid var(--border);border-radius:999px;background:var(--panel);font-size:12px;cursor:pointer;user-select:none}
.chip .sw{width:10px;height:10px;border-radius:2px;display:inline-block}
.chip.off{opacity:.45}
.chip small{color:var(--ink3)}
.grid{display:grid;grid-template-columns:minmax(380px,1fr) minmax(420px,1.1fr);gap:18px;margin-top:18px}
@media (max-width:980px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.card + .card{margin-top:18px}
svg{display:block;max-width:100%}
.pt{cursor:default}
.pt.sel{cursor:pointer}
.tip{position:fixed;pointer-events:none;background:#111;color:#fff;font-size:12px;padding:6px 8px;border-radius:6px;opacity:0;transition:opacity .08s;z-index:10;white-space:nowrap}
.tip b{color:#fff}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--ink2);margin:6px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:5px}
.legend i{width:9px;height:9px;border-radius:50%;display:inline-block}
.bar{cursor:pointer}
.bar.dim rect{opacity:.35}
.viewer{position:relative;background:#f7f7f5;border:1px solid var(--border);border-radius:8px;overflow:hidden}
canvas{display:block;width:100%;height:100%;cursor:grab}
canvas:active{cursor:grabbing}
.ctrls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font-size:12px;color:var(--ink2);margin:10px 0 0}
.ctrls label{display:inline-flex;align-items:center;gap:5px;cursor:pointer}
.ctrls select,.ctrls input[type=range]{font:inherit}
.ctrls button{font:inherit;font-size:12px;padding:3px 9px;border:1px solid var(--border);background:var(--panel);border-radius:6px;cursor:pointer}
.ctrls button:hover{background:#f2f1ee}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-size:13px;margin:8px 0}
.kv dt{color:var(--ink3)}
.kv dd{margin:0}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{font-weight:600;color:var(--ink2);text-align:left;padding:4px 6px;border-bottom:1px solid var(--border)}
td{padding:4px 6px;border-bottom:1px solid var(--grid);vertical-align:middle}
tr.row{cursor:pointer}
tr.row:hover td{background:#f5f4f1}
tr.row.cur td{background:#eef4fd}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.mini{height:8px;border-radius:4px;background:var(--s1);display:inline-block;vertical-align:middle}
.cbar{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink2)}
.cbar .ramp{width:120px;height:10px;border-radius:5px;background:linear-gradient(90deg,var(--seq100),var(--seq700))}
.note{font-size:12px;color:var(--ink3);margin-top:8px}
.hint{position:absolute;left:10px;bottom:8px;font-size:11px;color:var(--ink3);pointer-events:none}
.badge{position:absolute;right:10px;top:8px;font-size:11px;color:var(--ink2);background:rgba(255,255,255,.85);padding:2px 7px;border-radius:5px;border:1px solid var(--border)}
</style>
</head>
<body>
<h1 id="title"></h1>
<p class="sub" id="sub1"></p>
<p class="sub" id="sub2"></p>
<div class="chips" id="members"></div>

<div class="grid">
  <div>
    <div class="card">
      <h2 id="scatterTitle">Campaign: committee force disagreement per structure</h2>
      <div class="legend" id="scatterLegend"></div>
      <svg id="scatter"></svg>
      <p class="note">Each dot is one proposed structure scored by every seated member; y is the largest per-atom standard deviation of forces across the committee. Ringed dots are the top-k selected for DFT labeling — click one to inspect it. Hover any dot for details.</p>
    </div>
    <div class="card">
      <h2>Where does the committee disagree?</h2>
      <svg id="arms"></svg>
      <p class="note">Mean disagreement per (seed, perturbation family) arm; the Curator samples arms proportional to this and escalates the winning arm's magnitude each round. Click a bar to highlight its structures above.</p>
    </div>
    <div class="card">
      <h2>Selected structures (label these with DFT first)</h2>
      <table id="seltab"><thead><tr><th>#</th><th>id</th><th>seed</th><th>perturbation</th><th class="num">σ(F) eV/Å</th><th class="num">ΔE meV/atom</th><th></th></tr></thead><tbody></tbody></table>
    </div>
  </div>
  <div>
    <div class="card">
      <h2 id="vtitle">Structure</h2>
      <div class="viewer" id="viewerWrap" style="height:440px">
        <canvas id="cv"></canvas>
        <div class="hint">drag to rotate · scroll to zoom · double-click to reset</div>
        <div class="badge" id="badge"></div>
      </div>
      <div class="ctrls">
        <label>color atoms by
          <select id="colorMode">
            <option value="element">element</option>
            <option value="sigma" selected>per-atom force disagreement σ(F)</option>
            <option value="dev">|F<sub>member</sub> − F<sub>mean</sub>| for member…</option>
          </select>
        </label>
        <select id="devMember" style="display:none"></select>
        <span class="cbar" id="cbar"><span class="ramp"></span><span id="cbarLabel"></span></span>
      </div>
      <div class="ctrls">
        <span>force arrows:</span>
        <span id="arrowToggles" style="display:inline-flex;gap:8px;flex-wrap:wrap"></span>
        <label><input type="checkbox" id="showMean"> committee mean</label>
        <label>scale <input type="range" id="arrowScale" min="0" max="3" step="0.05" value="0.5"></label>
        <label><input type="checkbox" id="spin" checked> spin</label>
        <button id="reset">reset view</button>
      </div>
      <dl class="kv" id="kv"></dl>
      <h2 style="margin-top:12px">Who's the outlier on this structure?</h2>
      <table id="memtab"><thead><tr><th>member</th><th>joined</th><th class="num">E − Ē (meV/atom)</th><th>RMS |F − F̄| (eV/Å)</th><th class="num">max</th></tr></thead><tbody></tbody></table>
      <p class="note">Per-member deviation from the committee-mean forces on the atoms of the structure shown. Consensus isn't truth — but a member that's consistently the odd one out is telling you something about its training lineage.</p>
    </div>
  </div>
</div>
<div class="tip" id="tip"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const SERIES = ['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'];
// Four validated hues = four perturbation *families* (thermal / elastic /
// atom-removal-or-motion / site-disorder); arms within a family differ by glyph.
const TRANSFORM_COLOR = {rattle:'#2a78d6', strain:'#eb6834', vacancy:'#1baf7a', delith:'#1baf7a', hop:'#1baf7a', swap:'#4a3aa7', antisite:'#4a3aa7'};
const TRANSFORM_SHAPE = {rattle:'circle', strain:'square', vacancy:'diamond', delith:'tridown', hop:'cross', swap:'triangle', antisite:'star'};
const TRANSFORM_LABEL = {rattle:'rattle (thermal)', strain:'strain (elastic)', vacancy:'vacancy', delith:'delith (remove Li)', hop:'Li hop', swap:'antisite swap', antisite:'Li/TM antisite'};
const memberColor = {}; D.members.forEach((m,i)=>memberColor[m]=SERIES[i%SERIES.length]);
const $ = id => document.getElementById(id);
const fmt = (x,d=3) => (x==null||isNaN(x))?'–':Number(x).toFixed(d);
const tip = $('tip');
function showTip(html, ev){ tip.innerHTML=html; tip.style.opacity=1; tip.style.left=(ev.clientX+12)+'px'; tip.style.top=(ev.clientY+12)+'px'; }
function hideTip(){ tip.style.opacity=0; }

// ---------- header ----------
$('title').textContent = D.title;
document.title = D.title;
const seedStr = Object.entries(D.seeds).map(([n,s])=>`${n} (${s.mp_id||'built-in'}, ${s.natoms} atoms)`).join(', ');
$('sub1').textContent = `${D.n_evaluated} structures scored over ${D.rounds} rounds · committee of ${D.members.length}` + (D.never_joined.length?` · never joined: ${D.never_joined.join(', ')}`:'') + (D.dropped.length?` · dropped: ${D.dropped.join(', ')}`:'');
$('sub2').textContent = `seed structures: ${seedStr}`;
D.members.forEach(m=>{
  const c=document.createElement('span'); c.className='chip';
  c.innerHTML=`<span class="sw" style="background:${memberColor[m]}"></span>${m} <small>${D.joined[m]?`joined round ${D.joined[m]}`:'founding member'}</small>`;
  c.title='toggle this member\'s force arrows'; c.onclick=()=>{ arrowsOn[m]=!arrowsOn[m]; syncArrowToggles(); draw(); };
  c.dataset.member=m; $('members').appendChild(c);
});

// ---------- scatter ----------
const hist = D.history;
const selIds = new Map(D.selected.map((s,i)=>[s.id,i]));
let highlightArm = null;
function drawScatter(){
  const W = Math.min(720, $('scatter').parentElement.clientWidth-8), H=300, m={l:52,r:14,t:10,b:38};
  const svg=$('scatter'); svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('width',W); svg.setAttribute('height',H);
  const rounds = D.rounds; const ymax = Math.max(0.05, ...hist.map(h=>h.qbc_force_std))*1.12;
  const x = r => m.l + (r-0.5)/rounds*(W-m.l-m.r), y = v => H-m.b - v/ymax*(H-m.t-m.b);
  let s='';
  // grid + axes
  const yt = niceTicks(0,ymax,5);
  yt.forEach(v=>{ s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#efeeea"/>`; s+=`<text x="${m.l-8}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#52514e">${v.toFixed(2)}</text>`;});
  for(let r=1;r<=rounds;r++) s+=`<text x="${x(r)}" y="${H-m.b+16}" text-anchor="middle" font-size="11" fill="#52514e">${r}</text>`;
  s+=`<text x="${(m.l+W-m.r)/2}" y="${H-6}" text-anchor="middle" font-size="11.5" fill="#52514e">campaign round</text>`;
  s+=`<text transform="translate(13,${(m.t+H-m.b)/2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="#52514e">max per-atom σ(F)  (eV/Å)</text>`;
  // joined markers
  D.members.forEach(mm=>{ const j=D.joined[mm]; if(j){ const xx=x(j)+(x(2)-x(1))/2; /* seated during round j → votes from round j+1 */ s+=`<line x1="${xx}" x2="${xx}" y1="${m.t}" y2="${H-m.b}" stroke="#bbb" stroke-dasharray="3 3"/><text x="${xx+4}" y="${m.t+10}" font-size="10" fill="#8a8984">${mm} joined</text>`;}});
  // points (deterministic jitter by index in round)
  // history rounds are 0-based; display 1-based
  const byRound={}; hist.forEach(h=>{(byRound[h.round]??=[]).push(h)});
  hist.forEach((h,i)=>{
    const grp=byRound[h.round]; const k=grp.indexOf(h); const jit=((k+0.5)/grp.length-0.5)*0.7;
    const cx=x(h.round+1+jit), cy=y(h.qbc_force_std); const col=TRANSFORM_COLOR[h.transform]||'#888';
    const isSel=selIds.has(h.id); const dim = highlightArm && !(highlightArm.seed===h.seed&&highlightArm.transform===h.transform);
    s+=`<g class="pt ${isSel?'sel':''}" data-i="${i}" opacity="${dim?0.18:1}">`;
    if(isSel) s+=`<circle cx="${cx}" cy="${cy}" r="9" fill="none" stroke="#0b0b0b" stroke-width="1.6"/>`;
    s+=shape(h.transform,cx,cy,4.5,col);
    s+=`<circle cx="${cx}" cy="${cy}" r="11" fill="transparent"/></g>`;
  });
  svg.innerHTML=s;
  svg.querySelectorAll('.pt').forEach(g=>{
    const h=hist[+g.dataset.i];
    g.addEventListener('mousemove',ev=>showTip(`<b>${h.id}</b> · ${h.seed} / ${h.transform} @ ${h.magnitude}<br>σ(F) = ${fmt(h.qbc_force_std)} eV/Å · ΔE = ${fmt(h.energy_spread_per_atom*1000,1)} meV/atom${selIds.has(h.id)?'<br><i>selected — click to inspect</i>':''}`,ev));
    g.addEventListener('mouseleave',hideTip);
    if(selIds.has(h.id)) g.addEventListener('click',()=>select(selIds.get(h.id)));
  });
  const leg=$('scatterLegend'); leg.innerHTML = Object.keys(TRANSFORM_COLOR).filter(t=>hist.some(h=>h.transform===t)).map(t=>`<span><svg width="14" height="14">${shape(t,7,7,4.5,TRANSFORM_COLOR[t])}</svg>${TRANSFORM_LABEL[t]||t}</span>`).join('') + `<span><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="none" stroke="#0b0b0b" stroke-width="1.5"/></svg>selected top-${D.selected.length}</span>`;
}
function shape(t,cx,cy,r,col){
  switch(TRANSFORM_SHAPE[t]){
    case 'square': return `<rect x="${cx-r}" y="${cy-r}" width="${2*r}" height="${2*r}" rx="1" fill="${col}"/>`;
    case 'diamond': return `<polygon points="${cx},${cy-r*1.2} ${cx+r*1.2},${cy} ${cx},${cy+r*1.2} ${cx-r*1.2},${cy}" fill="${col}"/>`;
    case 'triangle': return `<polygon points="${cx},${cy-r*1.25} ${cx+r*1.15},${cy+r*0.85} ${cx-r*1.15},${cy+r*0.85}" fill="${col}"/>`;
    case 'tridown': return `<polygon points="${cx},${cy+r*1.25} ${cx+r*1.15},${cy-r*0.85} ${cx-r*1.15},${cy-r*0.85}" fill="${col}"/>`;
    case 'cross': return `<path d="M${cx-r*1.2},${cy}h${r*2.4}M${cx},${cy-r*1.2}v${r*2.4}" stroke="${col}" stroke-width="2.2" fill="none"/>`;
    case 'star': { let pts=[]; for(let k=0;k<10;k++){const rr=k%2?r*0.55:r*1.35; const a=-Math.PI/2+k*Math.PI/5; pts.push(`${cx+rr*Math.cos(a)},${cy+rr*Math.sin(a)}`);} return `<polygon points="${pts.join(' ')}" fill="${col}"/>`; }
    default: return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}"/>`;
  }
}
function niceTicks(a,b,n){ const span=b-a, raw=span/n, p=Math.pow(10,Math.floor(Math.log10(raw))), f=raw/p, step=(f<1.5?1:f<3?2:f<7?5:10)*p; const t=[]; for(let v=Math.ceil(a/step)*step; v<=b+1e-9; v+=step) t.push(+v.toFixed(6)); return t; }

// ---------- arms ----------
function drawArms(){
  const arms=[...D.arms].filter(a=>a.n>0).sort((a,b)=>b.mean_qbc-a.mean_qbc).slice(0,14);
  const W=Math.min(720,$('arms').parentElement.clientWidth-8), rowH=20, m={l:118,r:90,t:4,b:22}, H=m.t+m.b+arms.length*rowH;
  const svg=$('arms'); svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('width',W); svg.setAttribute('height',H);
  const xmax=Math.max(...arms.map(a=>a.mean_qbc))*1.05; const x=v=>m.l+v/xmax*(W-m.l-m.r);
  let s='';
  niceTicks(0,xmax,4).forEach(v=>{ s+=`<line x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${H-m.b}" stroke="#efeeea"/><text x="${x(v)}" y="${H-m.b+13}" text-anchor="middle" font-size="10.5" fill="#52514e">${v.toFixed(2)}</text>`;});
  arms.forEach((a,i)=>{
    const yy=m.t+i*rowH; const col=TRANSFORM_COLOR[a.transform]||'#888';
    const dim=highlightArm&&!(highlightArm.seed===a.seed&&highlightArm.transform===a.transform);
    s+=`<g class="bar ${dim?'dim':''}" data-i="${i}"><rect x="${m.l}" y="${yy+4}" width="${Math.max(2,x(a.mean_qbc)-m.l)}" height="${rowH-8}" rx="3" fill="${col}"/>`;
    s+=`<text x="${m.l-8}" y="${yy+rowH/2+4}" text-anchor="end" font-size="11.5" fill="#0b0b0b">${a.seed} · ${a.transform}</text>`;
    s+=`<text x="${x(a.mean_qbc)+6}" y="${yy+rowH/2+4}" font-size="11" fill="#52514e">${a.mean_qbc.toFixed(3)} <tspan fill="#8a8984">(n=${a.n}, → ${a.final_magnitude})</tspan></text></g>`;
  });
  svg.innerHTML=s;
  svg.querySelectorAll('.bar').forEach(g=>{ const a=arms[+g.dataset.i]; g.onclick=()=>{ highlightArm=(highlightArm&&highlightArm.seed===a.seed&&highlightArm.transform===a.transform)?null:a; drawScatter(); drawArms(); }; });
}

// ---------- selected table ----------
function drawTable(){
  const tb=$('seltab').querySelector('tbody'); const mx=Math.max(...D.selected.map(s=>s.qbc));
  tb.innerHTML=D.selected.map((s,i)=>`<tr class="row ${i===cur?'cur':''}" data-i="${i}"><td>${i+1}</td><td class="mono">${s.id}</td><td>${s.seed}</td><td>${s.transform} @ ${s.magnitude}</td><td class="num">${fmt(s.qbc)}</td><td class="num">${fmt(s.dE*1000,1)}</td><td><span class="mini" style="width:${Math.max(3,60*s.qbc/mx)}px"></span></td></tr>`).join('');
  tb.querySelectorAll('tr').forEach(tr=>tr.onclick=()=>select(+tr.dataset.i));
}

// ---------- 3D viewer ----------
const cv=$('cv'), ctx=cv.getContext('2d');
let cur=0, rot=[[1,0,0],[0,1,0],[0,0,1]], zoom=1, arrowsOn={}, spinning=true, dragging=false, last=null;
D.members.forEach(m=>arrowsOn[m]=false);
if (D.members.length) arrowsOn[D.members[0]] = true;
if (D.members.length>1) arrowsOn[D.members[1]] = true;
function mul(A,B){ const C=[[0,0,0],[0,0,0],[0,0,0]]; for(let i=0;i<3;i++)for(let j=0;j<3;j++)for(let k=0;k<3;k++)C[i][j]+=A[i][k]*B[k][j]; return C; }
function rotX(a){const c=Math.cos(a),s=Math.sin(a);return [[1,0,0],[0,c,-s],[0,s,c]];}
function rotY(a){const c=Math.cos(a),s=Math.sin(a);return [[c,0,s],[0,1,0],[-s,0,c]];}
function apply(R,v){ return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2], R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2], R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]; }
function resetView(){ rot=mul(rotX(-0.35),rotY(0.6)); zoom=1; }
resetView();

function seqColor(t){ // blue sequential ramp 100->700, t in [0,1]
  const a=[0xcd,0xe2,0xfb], b=[0x0d,0x36,0x6b]; t=Math.max(0,Math.min(1,t));
  return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`;
}
function shade(hex,f){ const n=parseInt(hex.slice(1),16); const r=(n>>16)&255,g=(n>>8)&255,b=n&255; return `rgb(${Math.round(r*f)},${Math.round(g*f)},${Math.round(b*f)})`; }

let derived=null; // per-structure computed values
function computeDerived(s){
  const N=s.numbers.length; const ms=D.members.filter(m=>s.member_forces[m]);
  const mean=Array.from({length:N},(_,i)=>[0,0,0]);
  ms.forEach(m=>s.member_forces[m].forEach((f,i)=>{mean[i][0]+=f[0]/ms.length;mean[i][1]+=f[1]/ms.length;mean[i][2]+=f[2]/ms.length;}));
  const dev={}; ms.forEach(m=>{ dev[m]=s.member_forces[m].map((f,i)=>Math.hypot(f[0]-mean[i][0],f[1]-mean[i][1],f[2]-mean[i][2])); });
  const stats=ms.map(m=>{ const d=dev[m]; const rms=Math.sqrt(d.reduce((a,v)=>a+v*v,0)/N); const mx=Math.max(...d);
    const eBar=ms.reduce((a,k)=>a+s.energies[k],0)/ms.length; return {m, rms, max:mx, dE:(s.energies[m]-eBar)/N*1000}; });
  const cen=[0,1,2].map(k=>s.cell[0][k]/2+s.cell[1][k]/2+s.cell[2][k]/2);
  const maxSigma=s.per_atom_sigma?Math.max(...s.per_atom_sigma):0;
  const maxDev=Math.max(1e-6,...ms.flatMap(m=>dev[m]));
  return {ms,mean,dev,stats,cen,maxSigma,maxDev};
}

function draw(){
  const s=D.selected[cur]; if(!s||!derived) return;
  const dpr=window.devicePixelRatio||1; const wrap=$('viewerWrap'); const W=wrap.clientWidth, H=wrap.clientHeight;
  if(cv.width!==W*dpr||cv.height!==H*dpr){cv.width=W*dpr;cv.height=H*dpr;}
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
  const d=derived, cell=s.cell;
  // scale: fit cell diagonal
  const diag=Math.hypot(...[0,1,2].map(k=>cell[0][k]+cell[1][k]+cell[2][k]));
  const sc=Math.min(W,H)/(diag*1.15)*zoom; const cx=W/2, cy=H/2;
  const P=v=>{const r=apply(rot,[v[0]-d.cen[0],v[1]-d.cen[1],v[2]-d.cen[2]]); return [cx+r[0]*sc, cy-r[1]*sc, r[2]];};
  // cell edges
  const o=[0,0,0], a=cell[0], b=cell[1], c=cell[2]; const add=(u,v)=>[u[0]+v[0],u[1]+v[1],u[2]+v[2]];
  const corners=[o,a,b,c,add(a,b),add(a,c),add(b,c),add(add(a,b),c)];
  const edges=[[0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7]];
  ctx.strokeStyle='#c9c8c2'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
  edges.forEach(([i,j])=>{const p=P(corners[i]),q=P(corners[j]); ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke();});
  ctx.setLineDash([]);
  // atoms
  const mode=$('colorMode').value; const devM=$('devMember').value;
  const items=s.numbers.map((z,i)=>({i,z,p:P(s.positions[i])})).sort((u,v)=>u.p[2]-v.p[2]);
  const arrowScale=+$('arrowScale').value*sc*0.9; // px per eV/Å (with user factor)
  const showMean=$('showMean').checked;
  items.forEach(({i,z,p})=>{
    const el=D.elements[z]; const r=el.radius*0.38*sc;
    let fill;
    if(mode==='element') fill=el.color;
    else if(mode==='sigma') fill=seqColor(s.per_atom_sigma?s.per_atom_sigma[i]/(d.maxSigma||1):0);
    else fill=seqColor(d.dev[devM]?d.dev[devM][i]/d.maxDev:0);
    const depth=(p[2]/(diag/2)+1)/2; // 0 back .. 1 front
    const g=ctx.createRadialGradient(p[0]-r*0.35,p[1]-r*0.35,r*0.1,p[0],p[1],r);
    g.addColorStop(0, mode==='element'?lighten(fill,0.35):fill); g.addColorStop(1, mode==='element'?shade(fill,0.7):fill);
    ctx.beginPath(); ctx.arc(p[0],p[1],r,0,Math.PI*2); ctx.fillStyle=g; ctx.globalAlpha=0.55+0.45*depth; ctx.fill(); ctx.globalAlpha=1;
    ctx.lineWidth=1; ctx.strokeStyle= mode==='element'?'rgba(0,0,0,.35)':'rgba(0,0,0,.25)'; ctx.stroke();
    // arrows for this atom
    const drawArrow=(f,col,w)=>{ const len=Math.hypot(f[0],f[1],f[2]); if(len*arrowScale<2) return; const q=apply(rot,f); const ex=p[0]+q[0]*arrowScale, ey=p[1]-q[1]*arrowScale;
      ctx.strokeStyle=col; ctx.fillStyle=col; ctx.lineWidth=w; ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(ex,ey); ctx.stroke();
      const ang=Math.atan2(ey-p[1],ex-p[0]); const hs=Math.min(7,3+w*1.5); ctx.beginPath(); ctx.moveTo(ex,ey); ctx.lineTo(ex-hs*Math.cos(ang-0.5),ey-hs*Math.sin(ang-0.5)); ctx.lineTo(ex-hs*Math.cos(ang+0.5),ey-hs*Math.sin(ang+0.5)); ctx.closePath(); ctx.fill(); };
    d.ms.forEach(m=>{ if(arrowsOn[m]) drawArrow(s.member_forces[m][i], memberColor[m], 1.8); });
    if(showMean) drawArrow(d.mean[i], '#0b0b0b', 2.4);
  });
  // element key
  let kx=10, ky=H-30; ctx.font='11px ui-sans-serif, sans-serif';
  Object.entries(D.elements).filter(([z])=>s.numbers.includes(+z)).forEach(([z,el])=>{ ctx.beginPath(); ctx.arc(kx+6,ky,6,0,Math.PI*2); ctx.fillStyle=mode==='element'?el.color:'#ddd'; ctx.fill(); ctx.strokeStyle='rgba(0,0,0,.35)'; ctx.stroke(); ctx.fillStyle='#52514e'; ctx.fillText(el.symbol,kx+16,ky+4); kx+=18+ctx.measureText(el.symbol).width+10; });
  ctx.fillStyle='#8a8984'; ctx.fillText(`${s.numbers.length} atoms`,kx+4,ky+4);
  // arrow legend (top-left)
  let ly=16; ctx.font='11px ui-sans-serif, sans-serif';
  d.ms.forEach(m=>{ if(!arrowsOn[m]) return; ctx.strokeStyle=memberColor[m]; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(10,ly); ctx.lineTo(30,ly); ctx.stroke(); ctx.fillStyle='#52514e'; ctx.fillText(m,36,ly+4); ly+=16; });
  if(showMean){ ctx.strokeStyle='#0b0b0b'; ctx.lineWidth=2.4; ctx.beginPath(); ctx.moveTo(10,ly); ctx.lineTo(30,ly); ctx.stroke(); ctx.fillStyle='#52514e'; ctx.fillText('committee mean',36,ly+4); }
}
function lighten(hex,f){ const n=parseInt(hex.slice(1),16); const r=(n>>16)&255,g=(n>>8)&255,b=n&255; return `rgb(${Math.round(r+(255-r)*f)},${Math.round(g+(255-g)*f)},${Math.round(b+(255-b)*f)})`; }

function select(i){
  cur=i; const s=D.selected[i]; derived=computeDerived(s);
  $('vtitle').textContent=`#${i+1} · ${s.id} — ${s.seed} / ${s.transform} @ ${s.magnitude}`;
  $('badge').textContent=`σ(F) ${fmt(s.qbc)} eV/Å · round ${s.round+1}`;
  const info=s.info||{}; const prov=info.mp_id?`${info.mp_id}${info.spacegroup?` (${info.spacegroup})`:''}`:'built-in mock seed';
  $('kv').innerHTML=`<dt>seed</dt><dd>${s.seed} — ${prov}</dd><dt>perturbation</dt><dd>${s.transform} at magnitude ${s.magnitude}</dd><dt>committee σ(F)</dt><dd>${fmt(s.qbc)} eV/Å (max per-atom)</dd><dt>energy spread</dt><dd>${fmt(s.dE*1000,1)} meV/atom across ${derived.ms.length} members</dd>`;
  const tb=$('memtab').querySelector('tbody'); const mx=Math.max(...derived.stats.map(t=>t.rms),1e-9);
  tb.innerHTML=derived.stats.sort((a,b)=>b.rms-a.rms).map(t=>`<tr><td><span class="sw" style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${memberColor[t.m]};margin-right:6px;vertical-align:middle"></span>${t.m}</td><td>${D.joined[t.m]?`r${D.joined[t.m]}`:'founding'}</td><td class="num">${t.dE>=0?'+':''}${fmt(t.dE,1)}</td><td><span class="mini" style="width:${Math.max(3,110*t.rms/mx)}px;background:${memberColor[t.m]}"></span> <span class="mono">${fmt(t.rms)}</span></td><td class="num">${fmt(t.max)}</td></tr>`).join('');
  const dm=$('devMember'); dm.innerHTML=derived.ms.map(m=>`<option value="${m}">${m}</option>`).join('');
  syncArrowToggles(); updateCbar(); drawTable(); drawScatter(); draw();
}
function syncArrowToggles(){
  const s=D.selected[cur]; const ms=D.members.filter(m=>s.member_forces[m]);
  $('arrowToggles').innerHTML=ms.map(m=>`<label><input type="checkbox" data-m="${m}" ${arrowsOn[m]?'checked':''}> <span class="sw" style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${memberColor[m]}"></span>${m}</label>`).join('');
  $('arrowToggles').querySelectorAll('input').forEach(inp=>inp.onchange=()=>{arrowsOn[inp.dataset.m]=inp.checked; syncArrowToggles(); draw();});
  document.querySelectorAll('#members .chip').forEach(c=>c.classList.toggle('off',!arrowsOn[c.dataset.member]));
}
function updateCbar(){
  const mode=$('colorMode').value; $('devMember').style.display=mode==='dev'?'':'none';
  if(mode==='element'){ $('cbar').style.visibility='hidden'; return; } $('cbar').style.visibility='visible';
  $('cbarLabel').textContent = mode==='sigma' ? `0 → ${fmt(derived.maxSigma)} eV/Å per-atom σ(F)` : `0 → ${fmt(derived.maxDev)} eV/Å |F − F̄|`;
}
$('colorMode').onchange=()=>{updateCbar();draw();}; $('devMember').onchange=draw; $('showMean').onchange=draw; $('arrowScale').oninput=draw;
$('spin').onchange=e=>{spinning=e.target.checked;}; $('reset').onclick=()=>{resetView();draw();};
cv.addEventListener('mousedown',e=>{dragging=true;last=[e.clientX,e.clientY];spinning=false;$('spin').checked=false;});
window.addEventListener('mouseup',()=>dragging=false);
window.addEventListener('mousemove',e=>{ if(!dragging) return; const dx=e.clientX-last[0], dy=e.clientY-last[1]; last=[e.clientX,e.clientY]; rot=mul(mul(rotX(dy*0.01),rotY(dx*0.01)),rot); draw(); });
cv.addEventListener('wheel',e=>{ e.preventDefault(); zoom*=Math.exp(-e.deltaY*0.0015); zoom=Math.max(0.3,Math.min(4,zoom)); draw(); },{passive:false});
cv.addEventListener('dblclick',()=>{resetView();draw();});
cv.addEventListener('touchstart',e=>{ if(e.touches.length===1){dragging=true;last=[e.touches[0].clientX,e.touches[0].clientY];spinning=false;$('spin').checked=false;} },{passive:true});
cv.addEventListener('touchmove',e=>{ if(!dragging||e.touches.length!==1) return; const t=e.touches[0]; const dx=t.clientX-last[0], dy=t.clientY-last[1]; last=[t.clientX,t.clientY]; rot=mul(mul(rotX(dy*0.01),rotY(dx*0.01)),rot); draw(); },{passive:true});
window.addEventListener('touchend',()=>dragging=false);
window.addEventListener('resize',()=>{drawScatter();drawArms();draw();});
drawArms(); if(D.selected.length){ select(0); } else { drawScatter(); }
(function tick(){ if(spinning){ rot=mul(rotY(0.006),rot); draw(); } requestAnimationFrame(tick); })();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", type=Path, help="committee_report.json from the campaign")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output HTML (default: committee_report.html beside the JSON)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    report = json.loads(args.report.read_text())
    if not any("member_forces" in c for c in report.get("selected", [])):
        print("warning: this report has no per-member forces on the selected "
              "structures (older campaign); force arrows and per-atom "
              "disagreement will be unavailable")
    data = build_data(report, args.title)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = HTML.replace("__TITLE__", data["title"]).replace("__DATA__", payload)
    out = args.out or args.report.with_name("committee_report.html")
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(data['selected'])} selected structures, "
          f"{len(data['history'])} scored)")


if __name__ == "__main__":
    main()
