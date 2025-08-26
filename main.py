import json
from service.ml.contextual_model_trainer import ContextualModelTrainer

LOCAL_TZ = 'Europe/Rome'
MIN_SCAN_COVERAGE_S = 30

if __name__ == "__main__":
    import argparse

    #
    parser = argparse.ArgumentParser(description="Calibrate ScanWatch HR to T10 using monotone models with context.")
    parser.add_argument("--scan_csv", required=True, help="ScanWatch CSV (start,duration,value arrays)")
    parser.add_argument("--sqlite", required=True, help="chroniax SQLite database")
    parser.add_argument("--user_id", type=int, default=None, help="Optional userId to filter")
    parser.add_argument("--freq", default="1min", help="Bin size, e.g., 1min or 30S")
    parser.add_argument("--model", default="pchip", choices=["pchip", "isotonic"], help="Calibrator type")
    parser.add_argument("--out_pairs", default="dataset.csv", help="Output CSV of paired data with predictions")
    parser.add_argument("--out_models", default="models.json", help="Output JSON of chosen models and metrics")
    args = parser.parse_args()

    trainer = ContextualModelTrainer(model_kind=args.model, local_tz=LOCAL_TZ, min_scan_coverage_s=MIN_SCAN_COVERAGE_S)
    pairs, summary = trainer.train_and_apply(args.scan_csv, args.sqlite, args.user_id, args.freq)

    pairs.to_csv(args.out_pairs, index=False)
    with open(args.out_models, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out_pairs} and {args.out_models}")
