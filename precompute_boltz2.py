#!/usr/bin/env python3
"""
PaperToProtein — Boltz-2 Pre-computation Script
Run this ONCE to generate real Boltz-2 predictions for your demo papers.
Saves results as JSON that you paste into the app's cached data.

Usage:
  pip install requests
  python precompute_boltz2.py

Takes ~5-15 min depending on how many candidates you validate.
"""

import requests
import json
import time
import sys
import os

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BOLTZ2_URL = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"

# Target sequences — real sequences from UniProt
# PD-L1 extracellular IgV domain (the part binders target)
TARGETS = {
    "PD-L1": {
        "uniprot": "Q9NZQ7",
        # PD-L1 IgV domain residues 18-134 (extracellular, where antibodies bind)
        "sequence": "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYNKINQRILVVDPVTSEHELTCQAEGYPKAEVIWTSSDHQVLSGKTTTTNSKREEKLFNVTSTL",
    },
    "EGFR": {
        "uniprot": "P00533",
        # EGFR domain III (residues 310-480, ligand binding domain)
        "sequence": "CPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMPIWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLTSIISAVVGILLVVVLGVVFGILIKRRQQKIRKYTMRRLLQETELVEPLTPSGAMPNQAQMRILKETELRKVKVLGSGAFGTVYKGIWIPDGENVKIPVAIKVLRENTSPKANKEILDEAYVMAGVGSPYVSRLLGICL",
    },
    "Spike-RBD": {
        "uniprot": "P0DTC2",
        # SARS-CoV-2 Spike RBD (residues 319-541)
        "sequence": "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF",
    },
}

# How many binder candidates to validate per target
# More = better data but slower. 5-10 for demo, 47 for full run.
NUM_CANDIDATES_PER_TARGET = 5  # Start with 5, increase if time permits


def generate_dummy_binder(seed, length=90):
    """Generate a plausible binder sequence"""
    import random
    random.seed(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(random.choice(aa) for _ in range(length))


def predict_complex(target_seq, binder_seq, design_id):
    """Call Boltz-2 to predict target+binder complex"""
    payload = {
        "polymers": [
            {"id": "A", "molecule_type": "protein", "sequence": target_seq},
            {"id": "B", "molecule_type": "protein", "sequence": binder_seq},
        ],
        "recycling_steps": 3,
        "sampling_steps": 50,
        "diffusion_samples": 1,
        "step_scale": 1.638,
        "output_format": "mmcif",
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
    }

    try:
        print(f"  [{design_id}] Calling Boltz-2...", end=" ", flush=True)
        start = time.time()
        resp = requests.post(BOLTZ2_URL, json=payload, headers=headers, timeout=300)
        elapsed = time.time() - start

        if resp.status_code != 200:
            print(f"ERROR {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        confidence = data.get("confidence_scores", [None])[0]
        structure = data.get("structures", [{}])[0].get("structure", None)

        # Binding affinity fields (may not be available in all NIM versions)
        ic50 = data.get("binding_affinity", {}).get("ic50") or data.get("ic50")
        likelihood = data.get("binding_affinity", {}).get("likelihood") or data.get("affinity_likelihood")

        print(f"OK ({elapsed:.1f}s) confidence={confidence}")
        return {
            "design_id": design_id,
            "confidence": confidence,
            "ic50": ic50,
            "affinity_likelihood": likelihood,
            "has_structure": structure is not None,
            "structure_length": len(structure) if structure else 0,
            "structure_mmcif": structure,
            "elapsed_seconds": round(elapsed, 1),
        }

    except requests.exceptions.Timeout:
        print("TIMEOUT (>300s)")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def test_connection():
    """Quick test that the API key works"""
    print("Testing Boltz-2 API connection...")
    payload = {
        "polymers": [
            {"id": "A", "molecule_type": "protein", "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQ"},
        ],
        "recycling_steps": 1,
        "sampling_steps": 10,
        "diffusion_samples": 1,
        "output_format": "mmcif",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
    }
    try:
        resp = requests.post(BOLTZ2_URL, json=payload, headers=headers, timeout=120)
        if resp.status_code == 200:
            print(f"  ✓ Connection OK! Response has {len(resp.json().get('structures', []))} structure(s)")
            return True
        else:
            print(f"  ✗ Failed: {resp.status_code} — {resp.text[:300]}")
            return False
    except Exception as e:
        print(f"  ✗ Connection error: {e}")
        return False


def main():
    print("=" * 60)
    print("PaperToProtein — Boltz-2 Pre-computation")
    print("=" * 60)
    print()

    # Test connection first
    if not test_connection():
        print("\nAPI connection failed. Check your NVIDIA_API_KEY.")
        print("Get one at: https://build.nvidia.com/settings/api-keys")
        sys.exit(1)

    print()
    all_results = {}

    for target_name, target_info in TARGETS.items():
        print(f"\n{'─' * 50}")
        print(f"Target: {target_name} ({len(target_info['sequence'])} residues)")
        print(f"{'─' * 50}")

        results = []
        for i in range(NUM_CANDIDATES_PER_TARGET):
            design_id = f"PTP_{i:03d}"
            binder_seq = generate_dummy_binder(seed=hash(target_name) + i * 137, length=80 + (i * 7) % 60)

            result = predict_complex(target_info["sequence"], binder_seq, design_id)
            if result:
                results.append(result)

            # Rate limiting — be nice to the API
            if i < NUM_CANDIDATES_PER_TARGET - 1:
                time.sleep(2)

        all_results[target_name] = {
            "target_sequence_length": len(target_info["sequence"]),
            "num_validated": len(results),
            "results": results,
        }

        # Print summary
        confidences = [r["confidence"] for r in results if r["confidence"] is not None]
        if confidences:
            print(f"\n  Summary: {len(results)} predictions")
            print(f"  Avg confidence: {sum(confidences)/len(confidences):.3f}")
            print(f"  Best confidence: {max(confidences):.3f}")
            print(f"  Worst confidence: {min(confidences):.3f}")

    # Save results
    output_file = "boltz2_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Results saved to {output_file}")
    print(f"{'=' * 60}")
    print()
    print("Next steps:")
    print("1. Open boltz2_results.json")
    print("2. Use the confidence scores to replace the cached")
    print("   boltz_confidence values in your App.jsx")
    print("3. If IC50 values came back, update boltz_ic50_uM too")


if __name__ == "__main__":
    main()
