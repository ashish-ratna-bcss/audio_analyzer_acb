"""CLI: score an evaluation dataset offline.

    python -m eval.run --dataset eval_data/ --out report.json

Reads each case's reference + hypothesis (see eval.reference_store), scores with
the ASR metrics, and writes a JSON report + prints a human-readable summary. The
domain glossary (for entity accuracy) is loaded from config when available.
"""
import argparse
import json
import sys

from eval import reference_store, runner


def _glossary():
    try:
        import config
        return getattr(config, "GLOSSARY", {}) or {}
    except Exception:
        return {}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline ASR evaluation harness")
    ap.add_argument("--dataset", required=True, help="dataset root (dir of case dirs)")
    ap.add_argument("--out", default="eval_report.json", help="JSON report path")
    args = ap.parse_args(argv)

    pairs = reference_store.load_dataset(args.dataset)
    if not pairs:
        print(f"no cases found under {args.dataset}", file=sys.stderr)
        return 1

    report = runner.evaluate(pairs, glossary=_glossary())
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    agg = report["aggregate"]
    print(f"\n=== ASR eval: {report['files_scored']} scored, "
          f"{report['files_errored']} errored ===")
    print(f"  WER {agg['wer']}  CER {agg['cer']}  SFR {agg['sfr']}  "
          f"num {agg['number_acc']}  entity {agg['entity_acc']}")
    if report["by_engine"]:
        print("  per engine (WER):")
        for eng, m in sorted(report["by_engine"].items()):
            print(f"    {eng:16s} WER {m['wer']}  CER {m['cer']}  SFR {m['sfr']}  segs {m['segments']}")
    if report["worst_files"]:
        print("  worst:", ", ".join(report["worst_files"][:5]))
    print(f"  full report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
