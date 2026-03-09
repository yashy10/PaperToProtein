"""
PaperToProtein — Modal Backend
Serves Boltz-2 protein complex predictions on GPU via web endpoints.
Deploy: modal deploy modal_backend.py
Dev:    modal serve modal_backend.py
"""

from pathlib import Path
from typing import Optional

import modal

MINUTES = 60  # seconds

app = modal.App(name="paper-to-protein")

# Container image with Boltz-2 and FastAPI
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "boltz==2.1.1",
    "fastapi[standard]",
)

# Persistent volume for cached Boltz-2 model weights (~5GB)
boltz_model_volume = modal.Volume.from_name("boltz-models", create_if_missing=True)
models_dir = Path("/models/boltz")


# ─────────────────────────────────────────────────────────────────────────────
# Model Download (run once, cached in volume)
# ─────────────────────────────────────────────────────────────────────────────

download_image = (
    modal.Image.debian_slim()
    .uv_pip_install("huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    volumes={models_dir: boltz_model_volume},
    timeout=20 * MINUTES,
    image=download_image,
)
def download_model(force_download: bool = False):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="boltz-community/boltz-2",
        revision="6fdef46d763fee7fbb83ca5501ccceff43b85607",
        local_dir=models_dir,
        force_download=force_download,
    )
    boltz_model_volume.commit()
    print("Model downloaded to volume")


# ─────────────────────────────────────────────────────────────────────────────
# Boltz-2 Inference on GPU
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=image,
    volumes={models_dir: boltz_model_volume},
    timeout=10 * MINUTES,
    gpu="H100",
)
def boltz_predict_single(target_seq: str, binder_seq: str) -> dict:
    """Run Boltz-2 on a single target+binder pair and return structured results."""
    import json
    import subprocess
    import tarfile
    import io
    import tempfile
    import os

    work_dir = tempfile.mkdtemp()

    # Write Boltz-2 input YAML
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {target_seq}
  - protein:
      id: B
      sequence: {binder_seq}
"""
    input_path = Path(work_dir) / "input.yaml"
    input_path.write_text(yaml_content)

    print(f"Running Boltz-2 prediction: target={len(target_seq)}aa, binder={len(binder_seq)}aa")

    try:
        result = subprocess.run(
            [
                "boltz", "predict", str(input_path),
                "--use_msa_server",
                "--cache", str(models_dir),
                "--out_dir", work_dir,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5 * MINUTES,
        )
        print(f"Boltz-2 stdout: {result.stdout[-500:]}")
    except subprocess.CalledProcessError as e:
        print(f"Boltz-2 failed: {e.stderr[-500:]}")
        return {"error": str(e.stderr[-200:]), "confidence": None, "ic50": None}

    # Parse outputs — look for CIF structure and affinity JSON
    output = {
        "structure": None,
        "format": "mmcif",
        "confidence": None,
        "ic50": None,
        "affinity_likelihood": None,
    }

    results_dir = Path(work_dir)
    # Find output files recursively
    cif_files = list(results_dir.rglob("*.cif"))
    json_files = list(results_dir.rglob("*.json"))

    if cif_files:
        output["structure"] = cif_files[0].read_text()
        print(f"Found structure: {cif_files[0]} ({len(output['structure'])} chars)")

    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
            if "confidence" in data or "confidence_score" in data:
                output["confidence"] = data.get("confidence") or data.get("confidence_score")
            if "ic50" in data:
                output["ic50"] = data["ic50"]
            if "affinity" in data or "binding_affinity" in data:
                aff = data.get("affinity") or data.get("binding_affinity", {})
                if isinstance(aff, dict):
                    output["ic50"] = output["ic50"] or aff.get("ic50")
                    output["affinity_likelihood"] = aff.get("likelihood")
                else:
                    output["affinity_likelihood"] = aff
        except Exception:
            pass

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Web Endpoints (called from frontend)
# ─────────────────────────────────────────────────────────────────────────────


@app.function(image=image)
@modal.fastapi_endpoint(method="GET", docs=True)
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "paper-to-protein", "gpu": "H100"}


@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=True)
def predict(data: dict):
    """
    Predict complex structure of target + binder protein.

    Expected body:
    {
        "target_sequence": "MKKLL...",
        "binder_sequence": "ACDEF...",
    }
    """
    target_seq = data.get("target_sequence", "")
    binder_seq = data.get("binder_sequence", "")

    if not target_seq or not binder_seq:
        return {"error": "Both target_sequence and binder_sequence are required"}

    if len(target_seq) > 2000 or len(binder_seq) > 2000:
        return {"error": "Sequences must be under 2000 residues each"}

    result = boltz_predict_single.remote(target_seq, binder_seq)
    return result


@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=True)
def batch_predict(data: dict):
    """
    Batch predict complex structures for multiple binder candidates.

    Expected body:
    {
        "target_sequence": "MKKLL...",
        "binders": [
            {"id": "PTP_001", "sequence": "ACDEF..."},
            {"id": "PTP_002", "sequence": "GHIKL..."},
        ]
    }
    """
    target_seq = data.get("target_sequence", "")
    binders = data.get("binders", [])

    if not target_seq:
        return {"error": "target_sequence is required"}
    if not binders:
        return {"error": "binders list is required"}
    if len(binders) > 50:
        return {"error": "Maximum 50 binders per batch"}

    # Fan out predictions in parallel across GPUs
    results = []
    handles = []
    for binder in binders:
        seq = binder.get("sequence", "")
        if seq:
            handle = boltz_predict_single.spawn(target_seq, seq)
            handles.append((binder.get("id", "unknown"), handle))

    for binder_id, handle in handles:
        try:
            result = handle.get()
            result["designId"] = binder_id
            results.append(result)
        except Exception as e:
            results.append({"designId": binder_id, "error": str(e)})

    return {"results": results, "total": len(results)}


@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=True)
def predict_from_paper(data: dict):
    """
    Full pipeline: take parsed paper info, fetch UniProt, run predictions.

    Expected body:
    {
        "uniprot_id": "Q9NZQ7",
        "binder_sequences": ["ACDEF...", "GHIKL..."],
    }
    """
    uniprot_id = data.get("uniprot_id", "")
    binder_seqs = data.get("binder_sequences", [])

    if not uniprot_id or not binder_seqs:
        return {"error": "uniprot_id and binder_sequences are required"}

    # Fetch target sequence from UniProt (done server-side to avoid CORS)
    import urllib.request
    import json

    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            uni_data = json.loads(resp.read())
        target_seq = uni_data.get("sequence", {}).get("value", "")
    except Exception as e:
        return {"error": f"Failed to fetch UniProt {uniprot_id}: {str(e)}"}

    if not target_seq:
        return {"error": f"No sequence found for {uniprot_id}"}

    # Fan out Boltz-2 predictions in parallel
    handles = []
    for i, seq in enumerate(binder_seqs[:50]):
        handle = boltz_predict_single.spawn(target_seq, seq)
        handles.append((f"PTP_{i:03d}", handle))

    results = []
    for binder_id, handle in handles:
        try:
            result = handle.get()
            result["designId"] = binder_id
            results.append(result)
        except Exception as e:
            results.append({"designId": binder_id, "error": str(e)})

    return {
        "target_sequence": target_seq[:50] + "...",
        "target_length": len(target_seq),
        "results": results,
        "total": len(results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint: download model
# ─────────────────────────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(force_download: bool = False):
    print("Downloading Boltz-2 model weights...")
    download_model.remote(force_download)
    print("Done! Deploy with: modal deploy modal_backend.py")
