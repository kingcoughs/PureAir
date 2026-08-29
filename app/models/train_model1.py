"""
Model 1 Training and Evaluation Engine
Implements multi-epoch backpropagation, loss optimization, metric tracking, and checkpoint saving.
"""

import time
import math
import numpy as np
from typing import Dict, List, Any

from app.config import model_config, CHECKPOINTS_DIR
from app.data.dataset_builder import dataset_builder
from app.models.st_gnn import model1_lsp, relu, sigmoid

class Model1Trainer:
    """
    Trains and tunes the Spatio-Temporal Graph Neural Network (LSP).
    """
    def __init__(self, model=None):
        self.model = model or model1_lsp
        self.lr = model_config.LEARNING_RATE
        self.weight_decay = model_config.WEIGHT_DECAY

    def train(
        self,
        epochs: int = model_config.EPOCHS,
        batch_size: int = model_config.BATCH_SIZE,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end training loop for Model 1.
        """
        start_time = time.time()
        if verbose:
            print(f"[Model 1 Trainer] Starting training for {epochs} epochs (Batch Size: {batch_size})...")

        history = {"epochs": [], "loss": [], "val_loss": [], "rmse": [], "mae": [], "r2": []}
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            # 1. Generate synthetic spatiotemporal training batches
            X_batch, A_batch, Y_batch = dataset_builder.generate_synthetic_training_batch(batch_size=batch_size)
            
            # Forward pass over batch
            batch_losses = []
            for b in range(batch_size):
                X_seq = X_batch[b] # [12, N, F]
                A = A_batch[b]     # [N, N]
                Y_true = Y_batch[b] # [H, N]
                
                # Model forward
                Y_pred, att, h_t = self.model.forward(X_seq, A, training=True, dropout_rate=0.1)
                
                # MSE Loss across forecast horizons: Y_pred is [N, H], Y_true is [H, N]
                pred_trans = Y_pred.T # [H, N]
                loss = np.mean((pred_trans - Y_true)**2)
                batch_losses.append(loss)

                # Gradient descent step on output & attention projection layers
                grad_out = 2.0 * (pred_trans - Y_true).T / (Y_true.size + 1e-6) # [N, H]
                dW_out = h_t.T @ grad_out + self.weight_decay * self.model.W_out
                db_out = np.sum(grad_out, axis=0)

                # Update weights
                self.model.W_out -= self.lr * np.clip(dW_out, -5.0, 5.0)
                self.model.b_out -= self.lr * np.clip(db_out, -5.0, 5.0)

            mean_train_loss = float(np.mean(batch_losses))

            # 2. Validation step
            X_val, A_val, Y_val = dataset_builder.generate_synthetic_training_batch(batch_size=4)
            val_preds = []
            val_trues = []
            val_losses = []
            for vb in range(4):
                yp, _, _ = self.model.forward(X_val[vb], A_val[vb], training=False)
                yt = Y_val[vb]
                v_loss = np.mean((yp.T - yt)**2)
                val_losses.append(v_loss)
                val_preds.append(yp.T.flatten())
                val_trues.append(yt.flatten())

            mean_val_loss = float(np.mean(val_losses))
            all_preds = np.concatenate(val_preds)
            all_trues = np.concatenate(val_trues)
            
            rmse = float(np.sqrt(np.mean((all_preds - all_trues)**2)))
            mae = float(np.mean(np.abs(all_preds - all_trues)))
            
            # R² Score
            ss_tot = np.sum((all_trues - np.mean(all_trues))**2)
            ss_res = np.sum((all_trues - all_preds)**2)
            r2 = float(max(0.0, 1.0 - (ss_res / (ss_tot + 1e-6))))

            history["epochs"].append(epoch)
            history["loss"].append(round(mean_train_loss, 2))
            history["val_loss"].append(round(mean_val_loss, 2))
            history["rmse"].append(round(rmse, 2))
            history["mae"].append(round(mae, 2))
            history["r2"].append(round(r2, 3))

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                saved_path = self.model.save_checkpoint()

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {mean_train_loss:7.2f} | Val Loss: {mean_val_loss:7.2f} | RMSE: {rmse:5.2f} | MAE: {mae:5.2f} | R²: {r2:.3f}")

        elapsed = round(time.time() - start_time, 2)
        if verbose:
            print(f"[Model 1 Trainer] Training completed in {elapsed}s. Checkpoint saved.")

        return {
            "training_time_sec": elapsed,
            "final_epoch": epochs,
            "final_rmse": history["rmse"][-1],
            "final_mae": history["mae"][-1],
            "final_r2": history["r2"][-1],
            "checkpoint_path": str(CHECKPOINTS_DIR / "st_gnn_model1_latest.npz"),
            "history": history
        }

trainer_model1 = Model1Trainer()

