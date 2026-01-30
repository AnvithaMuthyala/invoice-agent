"""
Batch test runner — runs the full pipeline on every sample invoice
and saves individual + summary results to the results/ folder.

Usage:
    python tests/run_all.py
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run

SAMPLES_DIR = Path(__file__).resolve().parent / "sample_invoices"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_all():
    images = sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )

    if not images:
        print(f"No images found in {SAMPLES_DIR}")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = []

    for img_path in images:
        name = img_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"{'='*60}")

        entry = {"file": img_path.name}

        try:
            result = run(str(img_path))
        except Exception:
            tb = traceback.format_exc()
            print(f"FAILED: {tb}")
            entry["error"] = tb
            summary.append(entry)
            continue

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            entry["error"] = result["error"]
            summary.append(entry)
            continue

        # Save individual result
        out_path = RESULTS_DIR / f"{name}_{timestamp}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"Saved: {out_path.name}")

        # Collect summary row
        ev = result.get("evaluation", {})
        fc = ev.get("factual_completeness", {})
        q = ev.get("quality", {})
        pc = ev.get("parsing_consistency", {})

        entry.update({
            "insights_count": len(result.get("workflow", {}).get("insights", [])),
            "factual_completeness": fc.get("score", "N/A"),
            "accuracy": fc.get("accuracy_score", "N/A"),
            "coverage": fc.get("completeness_score", "N/A"),
            "quality": q.get("score", "N/A"),
            "parsing_consistency": pc.get("score", "N/A"),
            "overall": ev.get("overall_score", "N/A"),
        })
        summary.append(entry)

        print(f"  Overall: {entry['overall']}/100  |  "
              f"FC: {entry['factual_completeness']}%  |  "
              f"Quality: {entry['quality']}/4  |  "
              f"PC: {entry['parsing_consistency']}%")

    # Save summary
    summary_path = RESULTS_DIR / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    # Print final table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'File':<25} {'Overall':>8} {'FC':>6} {'Qual':>6} {'PC':>6}")
    print("-" * 55)
    for row in summary:
        if "error" in row:
            print(f"{row['file']:<25} {'ERROR':>8}")
        else:
            print(f"{row['file']:<25} {row['overall']:>8} {row['factual_completeness']:>6} {row['quality']:>6} {row['parsing_consistency']:>6}")

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run_all()
