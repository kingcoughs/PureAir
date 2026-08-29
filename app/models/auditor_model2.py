"""
Model 2 Auditor Batch Pipeline & Closed-Loop Retraining
Runs weekly audit, logs residuals, evaluates attribution, and fine-tunes Model 1.
"""

import time
import numpy as np
from typing import Dict, Any

from app.data.dataset_builder import dataset_builder
from app.models.blame_engine import model2_auditor
from app.models.train_model1 import trainer_model1

class AuditorBatchRunner:
    """
    Orchestrates the weekly audit and closed-loop continuous learning cycle.
    """
    def __init__(self):
        self.auditor = model2_auditor
        self.trainer = trainer_model1

    def run_weekly_audit(self) -> Dict[str, Any]:
        """Runs complete Integrated Gradients attribution and generates GRAP brief."""
        X_curr, A_curr = dataset_builder.build_current_node_features()
        report = self.auditor.generate_weekly_audit_report(X_curr, A_curr)
        return report

    def trigger_closed_loop_retraining(self, epochs: int = 10) -> Dict[str, Any]:
        """
        Executes active fine-tuning of Model 1 using recent residual errors.
        """
        audit_before = self.run_weekly_audit()
        train_result = self.trainer.train(epochs=epochs, batch_size=8, verbose=False)
        audit_after = self.run_weekly_audit()

        return {
            "retraining_timestamp": time.time(),
            "epochs_trained": epochs,
            "pre_training_rmse": audit_before["model1_performance_metrics"]["rmse_aqi_points"],
            "post_training_rmse": train_result["final_rmse"],
            "rmse_improvement_pts": round(audit_before["model1_performance_metrics"]["rmse_aqi_points"] - train_result["final_rmse"], 2),
            "final_r2": train_result["final_r2"],
            "status": "Model 1 Successfully Retrained and Calibrated"
        }

auditor_batch_runner = AuditorBatchRunner()

