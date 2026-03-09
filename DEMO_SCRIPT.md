# PaperToProtein — Demo Rehearsal Script

## Bio × AI Hackathon 2026 | YC HQ | March 8

---

## Pre-Demo Checklist (5:30 PM)

- [ ] Laptop charged, power cable accessible
- [ ] Chrome/Firefox open with app loaded at localhost
- [ ] WiFi tested — confirm API calls work (UniProt, Claude)
- [ ] Cached data verified — all 3 demo papers load correctly
- [ ] Screen resolution set to 1920×1080 or closest
- [ ] Browser zoom at 100%, no extensions showing
- [ ] Close Slack, email, notifications — Do Not Disturb ON
- [ ] Demo paper PDF ready on desktop (for the "drop" moment)
- [ ] Backup: mobile hotspot ready if WiFi fails
- [ ] Backup: screenshot/video of full demo flow saved locally

---

## The Demo (2 minutes, timed)

### ACT 1: The Hook (0:00 – 0:15)

**WHO SPEAKS:** Lead presenter

**SAY THIS:**

> "When a scientist reads a paper about a new drug target, it takes days to go from reading to running computational experiments — finding the protein, predicting its structure, designing binders, interpreting results. We built an AI agent that does it in minutes. And instead of giving you a spreadsheet, it gives you this."

**DO THIS:** Gesture to screen. App is on the Upload View. The particle animation and gradient title should be visible.

---

### ACT 2: The Drop (0:15 – 0:30)

**SAY THIS:**

> "It starts with a paper."

**DO THIS:** Drag the PDF from your desktop onto the upload zone. The glow animation fires.

> "Claude reads the paper, identifies the target protein — in this case, PD-L1, a major immune checkpoint in cancer — and kicks off the full design pipeline autonomously."

**ON SCREEN:** Pipeline view appears. Steps start ticking off with spinners.

**KEY MOMENT:** When step 2 completes, point to the green UniProt card:

> "That's live data — we just pulled the real protein sequence from UniProt. 290 amino acids."

Let 2-3 more steps tick. Don't wait for all 7.

---

### ACT 3: The Switch (0:30 – 0:45)

**SAY THIS:**

> "The full pipeline takes a few minutes with AlphaFold, RFdiffusion, and Boltz-2 validation. Let me show you what came back — 47 designed binder proteins, each validated by NVIDIA's Boltz-2 for complex structure prediction and binding affinity."

**DO THIS:** If pipeline hasn't finished, click the demo button for pre-loaded results. The Explorer View appears with the force graph animating in.

**PAUSE** for 2 seconds. Let the graph land. The visual impact matters.

---

### ACT 4: The Payoff — Graph Exploration (0:45 – 1:40)

This is where you spend the most time. This is the product.

**SAY THIS:**

> "Every dot is a designed protein. Edges connect structurally similar designs. They're colored by structural family — five distinct families emerged from clustering."

**DO THIS:** Point to distinct clusters in the graph.

> "Watch what happens when I switch to Boltz-2 Confidence."

**DO THIS:** Change the Color dropdown to "Boltz-2 Confidence". Nodes recolor.

> "Now I can see which designs Boltz-2 is most confident about. This is an independent validation — Boltz-2 predicts the actual target-binder complex structure and computes binding affinity. When two models agree, that's the strongest signal."

**DO THIS:** Click on a node in the densest cluster. Detail panel opens with 3D viewer.

> "Here's candidate PTP_028 — 9.1 kcal/mol affinity but only 72 pLDDT stability. The 3D structure shows a mixed alpha-beta fold."

**DO THIS:** Now click an outlier node, visually separated from the main clusters.

> "But look at this outlier. Moderate affinity, but 96 pLDDT stability and top designability. A spreadsheet sorted by affinity buries this at rank 19. The graph makes it obvious."

**DO THIS:** Click the "Summary" tab on the right panel.

> "And we generate a full campaign summary — every cluster characterized, a Pareto frontier identified, and three candidates recommended for experimental validation."

---

### ACT 5: The Close (1:40 – 2:00)

**SAY THIS:**

> "From paper to designed proteins to informed selection — fully autonomous. Claude orchestrates, Tamarind designs, Boltz-2 validates, Modal scales. Three AI models, one pipeline, zero spreadsheets."

**DO THIS:** Switch back to "Narrator" tab. Point to Claude's analysis.

> "This is how AI should augment scientists. Not replace their judgment — amplify it."

**END.**

---

## Fallback Scenarios

### Scenario A: WiFi dies during demo

1. App is already loaded locally — continue normally
2. Pipeline animation plays from cache regardless
3. UniProt card won't appear — skip that callout
4. Claude narration uses fallback text — works fine
5. Everything else is client-side

### Scenario B: Pipeline animation hangs

1. Say: "While that runs, let me show you the results"
2. Click any of the 3 cached demo paper buttons
3. Continue with Act 4

### Scenario C: Graph doesn't render

1. Refresh the page
2. Click "PD-L1" demo button
3. If still broken: show screenshots from backup

### Scenario D: Judge asks a hard question

**"Are these designs actually viable?"**
> "Every candidate goes through a three-model validation pipeline: RFdiffusion designs the backbone, ProteinMPNN sequences it, and then Boltz-2 independently predicts the full target-binder complex structure and computes a binding affinity likelihood. When all three models agree on a candidate, that's a much stronger computational signal than any single score. The real value is the exploration interface — instead of blindly trusting one metric, a scientist can see where the models agree and disagree."

**"How is this different from just using RFdiffusion directly?"**
> "RFdiffusion gives you backbones and numbers. We add two layers: Boltz-2 provides independent structural validation of each complex — it actually predicts how the binder sits against the target — and the Design Space Explorer lets you see the full landscape. It's the difference between getting search results and getting a map with reviews."

**"What about the biology — is PD-L1 a good test case?"**
> "PD-L1 is one of the most validated immune checkpoint targets in oncology — approved antibodies like atezolizumab already target it. We chose it because it's well-characterized, so judges and biologists can evaluate our pipeline against known data. The system works for any target — we also have EGFR and SARS-CoV-2 Spike demos ready."

**"What is Boltz-2 and why did you add it?"**
> "Boltz-2 is NVIDIA's structural biology foundation model — it predicts biomolecular complex structures and binding affinities. It's the first deep learning model approaching the accuracy of free energy perturbation methods while being 1000x faster. We use it as an orthogonal validation layer — if RFdiffusion says a design is good AND Boltz-2 independently confirms it binds, that's much more reliable than either model alone."

---

## Timing Rehearsal Log

| Run | Time | Notes |
|-----|------|-------|
| 1 | ___:___ | |
| 2 | ___:___ | |
| 3 | ___:___ | |
| 4 | ___:___ | |
| 5 | ___:___ | |

**Target: Under 2:00. Ideal: 1:45–1:55.**

---

## Team Roles During Demo

| Role | Person | Responsibility |
|------|--------|----------------|
| Presenter | ___ | Speaks the script, controls the mouse |
| Backup | ___ | Watches timer, ready to jump in if tech fails |
| Tech | ___ | Monitors terminal, ready to restart app |

---

## Night-Before Setup Checklist

- [ ] Run `npm install` and verify all dependencies
- [ ] Test all 3 demo papers load correctly
- [ ] Verify Claude API narration works (or fallback loads)
- [ ] Verify UniProt live fetch works
- [ ] Record a backup video of the full demo (screen recording)
- [ ] Take 5 screenshots of key moments for submission
- [ ] Write the submission description (keep on clipboard)
- [ ] Charge all devices
- [ ] Set 3 alarms for morning

---

*The cached data is not the backup plan. It IS the plan. The live pipeline is the bonus.*
