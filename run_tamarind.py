#!/usr/bin/env python3
"""
PaperToProtein — Tamarind Bio Integration
Runs the real protein design pipeline:
  1. AlphaFold: predict PD-L1 target structure
  2. Protein Design (RFdiffusion + ProteinMPNN): design binders
  3. Poll for results and download

Usage:
  python run_tamarind.py

After completion, results are saved locally and can be fed into Boltz-2.
"""

import requests
import json
import time
import sys
import os

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

TAMARIND_KEY = os.environ.get("TAMARIND_API_KEY", "")
BASE_URL = "https://app.tamarind.bio/api/"
HEADERS = {"x-api-key": TAMARIND_KEY}

# PD-L1 IgV domain — the extracellular region where binders target
PDL1_SEQUENCE = "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYNKINQRILVVDPVTSEHELTCQAEGYPKAEVIWTSSDHQVLSGKTTTTNSKREEKLFNVTSTL"


def submit_job(job_name, job_type, settings):
    """Submit a job to Tamarind"""
    params = {
        "jobName": job_name,
        "type": job_type,
        "settings": settings,
    }
    print(f"  Submitting {job_type} job: {job_name}...")
    resp = requests.post(BASE_URL + "submit-job", headers=HEADERS, json=params)
    print(f"  Response ({resp.status_code}): {resp.text[:200]}")
    return resp.status_code == 200


def get_jobs():
    """List all jobs"""
    resp = requests.get(BASE_URL + "jobs", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return []


def get_job_status(job_name):
    """Check status of a specific job"""
    resp = requests.get(BASE_URL + "jobs", headers=HEADERS, params={"jobName": job_name})
    if resp.status_code == 200:
        jobs = resp.json()
        if isinstance(jobs, list):
            for j in jobs:
                if j.get("jobName") == job_name or j.get("name") == job_name:
                    return j.get("status", "unknown")
        elif isinstance(jobs, dict):
            return jobs.get("status", "unknown")
    return "unknown"


def download_result(job_name, output_dir="."):
    """Download job results"""
    resp = requests.post(BASE_URL + "result", headers=HEADERS, json={"jobName": job_name})
    if resp.status_code == 200:
        # Save result
        content_type = resp.headers.get("content-type", "")
        if "json" in content_type:
            data = resp.json()
            path = f"{output_dir}/{job_name}_result.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved JSON result to {path}")
            return data
        else:
            # Binary file (PDB, tar, etc.)
            ext = "tar.gz" if "gzip" in content_type else "pdb"
            path = f"{output_dir}/{job_name}_result.{ext}"
            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"  Saved binary result to {path} ({len(resp.content)} bytes)")
            return path
    else:
        print(f"  Download failed ({resp.status_code}): {resp.text[:200]}")
        return None


def poll_until_done(job_name, timeout=600, interval=15):
    """Poll job status until complete or timeout"""
    start = time.time()
    while time.time() - start < timeout:
        status = get_job_status(job_name)
        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] Job '{job_name}' status: {status}")

        if status in ("completed", "succeeded", "done", "success"):
            return True
        if status in ("failed", "error", "cancelled"):
            print(f"  Job failed with status: {status}")
            return False

        time.sleep(interval)

    print(f"  Timeout after {timeout}s")
    return False


def test_connection():
    """Verify API key works"""
    print("Testing Tamarind API connection...")
    try:
        resp = requests.get(BASE_URL + "tools", headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            print(f"  ✓ Connected! Available tools received.")
            return True
        else:
            print(f"  ✗ Status {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Connection error: {e}")
        return False


def main():
    print("=" * 60)
    print("PaperToProtein — Tamarind Bio Pipeline")
    print("=" * 60)

    if not test_connection():
        print("\nAPI connection failed. Check your key.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────
    # STEP 1: AlphaFold — Predict PD-L1 target structure
    # ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("STEP 1: AlphaFold Structure Prediction")
    print(f"{'─' * 50}")
    print(f"Sequence: PD-L1 IgV domain ({len(PDL1_SEQUENCE)} residues)")

    af_job = "ptp_pdl1_alphafold"
    submit_job(af_job, "alphafold", {
        "sequence": PDL1_SEQUENCE,
        "numModels": "1",        # Just 1 model for speed
        "numRecycles": 3,
        "modelType": "auto",
    })

    print("\nPolling for AlphaFold results (may take 2-10 min)...")
    af_done = poll_until_done(af_job, timeout=600, interval=20)

    if af_done:
        print("AlphaFold complete! Downloading structure...")
        af_result = download_result(af_job)
    else:
        print("AlphaFold didn't complete in time. Continuing with Protein Design anyway...")

    # ─────────────────────────────────────────────────────────
    # STEP 2: Protein Design (RFdiffusion + ProteinMPNN)
    # ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("STEP 2: Protein Design (RFdiffusion + ProteinMPNN)")
    print(f"{'─' * 50}")

    # The "protein-design" type on Tamarind runs the full pipeline:
    # RFdiffusion → ProteinMPNN → AlphaFold validation
    # It requires a PDB file as input.
    #
    # Option A: Use the AlphaFold result from Step 1
    # Option B: Use a known PDB from the database
    #
    # For the hackathon demo, we'll try both approaches:

    # First, try using AlphaFold output if available
    if af_done and af_result:
        pdb_input = f"{af_job}/output.pdb"  # Reference previous job output
        print(f"Using AlphaFold output: {pdb_input}")
    else:
        # Download PD-L1 PDB from RCSB directly and upload
        print("Downloading PD-L1 structure from RCSB (4ZQK)...")
        pdb_resp = requests.get("https://files.rcsb.org/download/4ZQK.pdb", timeout=60)
        if pdb_resp.status_code == 200:
            with open("4ZQK.pdb", "wb") as f:
                f.write(pdb_resp.content)
            print(f"  Downloaded 4ZQK.pdb ({len(pdb_resp.content)} bytes)")

            # Upload to Tamarind
            print("  Uploading to Tamarind...")
            up_resp = requests.put(
                BASE_URL + "upload/4ZQK.pdb",
                headers={**HEADERS, "Content-Type": "application/octet-stream"},
                data=pdb_resp.content
            )
            print(f"  Upload response ({up_resp.status_code}): {up_resp.text[:200]}")
            pdb_input = "4ZQK.pdb"
        else:
            print("  Failed to download PDB. Cannot proceed with design.")
            sys.exit(1)

    # Submit protein design job
    design_job = "ptp_pdl1_binder_design"
    design_settings = {
        "pdbFile": pdb_input,
        # Hotspot residues on PD-L1 IgV domain binding face
        # These are key interface residues from PD-1/PD-L1 crystal structures
        "hotspotResidues": "A54 A56 A66 A68 A113 A115 A116 A117 A121 A122 A123 A124",
        "numDesigns": 10,           # 10 designs for hackathon speed
        "binderLength": "60-100",   # Binder size range
    }

    # Try the protein-design endpoint (combined RFdiffusion+MPNN+AF2)
    submit_job(design_job, "protein-design", design_settings)

    print("\nPolling for design results (may take 5-20 min)...")
    design_done = poll_until_done(design_job, timeout=1200, interval=30)

    if design_done:
        print("\nProtein design complete! Downloading results...")
        design_result = download_result(design_job, output_dir=".")
        print("\n✓ Binder designs ready!")
    else:
        print("\nDesign didn't complete in time.")
        print("Check your Tamarind dashboard: https://app.tamarind.bio/jobs")

    # ─────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Pipeline Summary")
    print(f"{'=' * 60}")
    print(f"AlphaFold: {'✓ Complete' if af_done else '✗ Pending'}")
    print(f"Design:    {'✓ Complete' if design_done else '✗ Pending'}")
    print()
    print("Check Tamarind dashboard for full results:")
    print("  https://app.tamarind.bio/jobs")
    print()
    print("Next: Feed designed binder sequences into Boltz-2")
    print("  python precompute_boltz2.py")


if __name__ == "__main__":
    main()
