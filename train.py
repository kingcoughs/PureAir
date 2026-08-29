"""
Standalone Training & Audit CLI Script for Delhi-NCR AI Air Quality Engine
Usage:
    python train.py --epochs 20 --batch-size 8 --run-audit
"""

import sys
import io
import argparse
import time

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from app.config import settings, CHECKPOINTS_DIR
from app.models.train_model1 import trainer_model1
from app.models.auditor_model2 import auditor_batch_runner

def main():
    parser = argparse.ArgumentParser(description="Train Model 1 ST-GNN and execute Model 2 Weekly Audit for Delhi-NCR.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs for Model 1 (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--run-audit", action="store_true", default=True, help="Run Model 2 weekly audit after training")

    args = parser.parse_args()

    print("=" * 75)
    print("   PUREAIR®: DELHI-NCR AIR QUALITY AI TRAINING & AUDIT ENGINE   ")
    print("=" * 75)

    print(f"\n[Phase 1] Training Model 1 (Live Spatiotemporal Predictor ST-GNN)...")
    train_results = trainer_model1.train(epochs=args.epochs, batch_size=args.batch_size, verbose=True)

    print("\n" + "-" * 75)
    print(" Model 1 Training Summary:")
    print(f"   - Epochs: {train_results['final_epoch']}")
    print(f"   - RMSE: {train_results['final_rmse']} AQI points")
    print(f"   - MAE: {train_results['final_mae']} AQI points")
    print(f"   - R^2 Accuracy: {train_results['final_r2']:.3f}")
    print(f"   - Checkpoint Saved: {train_results['checkpoint_path']}")
    print("-" * 75)

    if args.run_audit:
        print(f"\n[Phase 2] Executing Model 2 (Causal Trend & Attribution Analyzer)...")
        audit_report = auditor_batch_runner.run_weekly_audit()
        print("\n" + "=" * 75)
        print("   MODEL 2 WEEKLY AUDIT & SOURCE APPORTIONMENT BRIEF   ")
        print("=" * 75)
        print(f"Executive Summary: {audit_report['executive_summary']}\n")
        print("Citywide Integrated Gradients Source Apportionment:")
        for category, pct in audit_report["citywide_source_apportionment"].items():
            print(f"  * {category:<35} : {pct:>5.1f}%")

        print("\nTop Impact Hotspot Enforcement Directives:")
        for h in audit_report["top_vulnerable_hotspots"]:
            print(f"  [{h['rank']}] {h['locality']:<25} ({h['zone']:<15}) | Primary: {h['primary_contributor']}")
            print(f"      -> Action: {h['recommended_enforcement']}")
        print("=" * 75)

if __name__ == "__main__":
    main()

