# PaperToProtein — Design Space Explorer

**Drop a paper. Get designed proteins. Explore the landscape.**

A React web app that turns a research paper (or a cached demo) into a **protein design campaign**: it extracts a target protein, runs an AI/structural pipeline, and lets you explore the resulting binder candidates in an interactive design-space graph with 3D structure viewing and AI narration.

Built for **Bio × AI Hackathon 2026**. Integrates **Claude**, **Tamarind Bio**, **UniProt**, **RCSB PDB**, **Boltz-2** (NVIDIA/Modal), and **BioRender**.

---

## What’s happening (high level)

1. **Upload** — You pick a demo paper (PD-L1, EGFR, Spike-RBD) or drop a PDF.
2. **Pipeline** — The app runs (or simulates) a multi-step pipeline: read paper → get sequence → predict structure → design binders → validate with Boltz-2 → build a design landscape.
3. **Explorer** — You see an interactive **force-directed graph** of design candidates (nodes = designs, edges = structural similarity). Click a node to see 3D structure (binder, target, or complex), metrics, and AI-generated narrative.

So in one sentence: **paper text → target protein → designed binders → graph + 3D + narration.**

---

## Pipeline steps (what the app does)

| Step | Label in UI | What it does |
|------|-------------|--------------|
| 1 | **Reading Paper** | Claude parses the paper and extracts JSON: target protein name, gene, UniProt ID, organism, disease, known PDBs, binding hints, etc. |
| 2 | **Fetching Sequence** | Fetches the protein sequence from **UniProt** (by ID or search). |
| 3 | **Tamarind AlphaFold** | **Tamarind Bio API**: AlphaFold structure prediction from sequence (or uses an existing PDB from the paper). |
| 4 | **Identifying Binding Sites** | Claude analyzes the structure and returns binding-site info (pockets, residues, druggability) for design. |
| 5 | **Tamarind RFdiffusion + ProteinMPNN** | **Tamarind**: RFdiffusion designs binder backbones; ProteinMPNN designs sequences. |
| 6 | **Boltz-2 via Modal GPU** | **Boltz-2** (NVIDIA NIM or Modal backend): predicts target–binder complex structure and binding affinity (IC50, confidence). |
| 7 | **Scoring & Ranking** | Combines pLDDT, Boltz-2 affinity, designability into rankings. |
| 8 | **Building Design Landscape** | Builds similarity edges (TMalign-style) and clusters; renders the force-directed graph. |

When you use **demo papers** (no PDF, no keys), steps are **simulated** with cached/synthetic data so you can see the full flow and explorer without calling external APIs.

---

## Tech stack

- **Frontend:** React 18, Vite 5
- **Graph:** D3 (force simulation, zoom, drag)
- **3D:** Three.js (protein backbone visualization, mmCIF/PDB parsing)
- **APIs used from the app:**  
  Claude (Anthropic), Tamarind Bio, UniProt (REST), RCSB PDB (files), Boltz-2 (NVIDIA health API or Modal), BioRender (link + figure description)

---

## How to run

```bash
npm install
npm run dev
```

App runs at **http://localhost:3001** (see `vite.config.js`). Build: `npm run build`; preview build: `npm run preview`.

---

## API keys and backends (optional)

The app works **without any keys** using demo papers and cached data. For live paper → design runs:

- **Claude (Anthropic)** — Paper parsing, binding-site analysis, landscape narration. Set your API key where the app calls the Anthropic API (e.g. in `App.jsx` or via env).
- **Tamarind Bio** — AlphaFold, RFdiffusion, ProteinMPNN. The app has a placeholder Tamarind API base and key; replace with your own.
- **Boltz-2** — Prefer **Modal** GPU backend (batch endpoint); falls back to **NVIDIA** cloud. Modal URLs are set in `Boltz2API` in `App.jsx`; NVIDIA uses an API key if you use the cloud endpoint.
- **BioRender** — No key in app; provides “Open in BioRender” and a figure description you can copy for BioRender’s AI.

Secrets/keys should not be committed; use environment variables or a secure config in production.

---

## Data and demos

- **Demo papers:** Three built-in campaigns — **PD-L1**, **EGFR**, **Spike-RBD** — with predefined targets and **synthetic** design sets (plus a few “real” PD-L1 designs).
- **Real PD-L1 designs:** The first five PD-L1 candidates use **real** metrics and Boltz-2–style outputs; their complex structures are in **`public/`** as mmCIF files (`design0_complex.cif` … `design4_complex.cif`, `pdl1_complex.cif`). The 3D viewer loads these for the “Target” and “Complex” tabs when a real design is selected.
- **Rest of the graph:** Other nodes are generated (seeded) to fill the design space and show clusters/families; edges are synthetic similarity links for the force graph.

---

## Main UI views

1. **Upload** — Drop zone for PDF + buttons for “run with cached demo paper” (PD-L1, EGFR, Spike-RBD).
2. **Pipeline** — Step-by-step progress (Reading Paper → … → Building Design Landscape) with short descriptions and optional live data (e.g. UniProt card, Tamarind/Boltz-2 status).
3. **Explorer** —  
   - **Left:** D3 force graph (color by cluster, affinity, stability, Boltz-2, etc.; adjustable edge threshold).  
   - **Right:** Detail panel (selected design), and tabs for **Narrator** (Claude markdown), **Summary** (campaign stats, families, top candidates, Boltz-2 badge), **BioRender** (link + figure description + copy).

The 3D viewer in the detail panel can show **Binder** (design only), **Target** (Boltz-2 target chain), or **Complex** (Boltz-2 target+binder); for real PD-L1 designs it uses the mmCIF files from `public/`.

---

## File layout (relevant to “what’s happening”)

- **`src/App.jsx`** — All logic: agent prompts, `AgentPipeline`, Tamarind/UniProt/PDB/Boltz-2 clients, demo data, pipeline steps, and React components (Upload, Pipeline, Explorer, ForceGraph, ProteinViewer, DetailPanel, NarrationPanel, BioRenderPanel, SummaryFigure).
- **`src/main.jsx`** — React root; mounts `App` into `#root`.
- **`index.html`** — Entry HTML; title “PaperToProtein — Design Space Explorer”.
- **`public/*.cif`** — Precomputed Boltz-2-style complex structures for the five real PD-L1 designs and a fallback; used by the 3D viewer.
- **`vite.config.js`** — Vite + React, dev server port 3001, open browser.
- **`package.json`** — Scripts and deps: `react`, `react-dom`, `d3`, `three`, `vite`, `@vitejs/plugin-react`.

---

## Summary

**PaperToProtein** turns a paper (or demo) into a **design campaign**: **Claude** + **UniProt** + **Tamarind** (AlphaFold, RFdiffusion, ProteinMPNN) + **Boltz-2** (complex + affinity). The result is an **interactive design-space explorer**: D3 force graph, 3D protein viewer (Three.js), and AI narration/summary/BioRender export. Run with `npm run dev`, use demo papers without keys, or add API keys to drive the full pipeline from a real PDF.
# PaperToProtein
