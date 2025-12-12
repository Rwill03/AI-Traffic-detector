"""
Evalueer het opgeslagen traffic model op de validatieset.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Zorg dat de project root (/app) op sys.path staat zodat transformer_model te importeren is
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from transformer_model.config import BEST_MODEL_PATH, get_device
from transformer_model.model import create_model
from transformer_model.data_preprocessing import prepare_data, load_scaler


def evaluate(verbose: bool = True):
    device = get_device()
    
    try:
        _, val_loader, scaler = prepare_data(reuse_scaler=True)
    except ValueError as e:
        print(f"Error tijdens voorbereiden data: {e}")
        return None
    
    if val_loader is None:
        print("Geen validatieset beschikbaar; verzamel meer data en train opnieuw.")
        return None
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Geen model gevonden op {BEST_MODEL_PATH}. Train het model eerst.")
        return None
    
    if scaler is None:
        print("Geen scaler gevonden. Train het model eerst zodat scaler.pkl wordt opgeslagen.")
        return None
    
    model = create_model(device)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            output = model(src)
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(tgt.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Unscale naar echte aantallen
    preds_unscaled = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    targets_unscaled = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    
    mae = mean_absolute_error(targets_unscaled.flatten(), preds_unscaled.flatten())
    mse = mean_squared_error(targets_unscaled.flatten(), preds_unscaled.flatten())
    rmse = float(np.sqrt(mse))
    mape = np.mean(np.abs((targets_unscaled - preds_unscaled) / (targets_unscaled + 1e-6))) * 100
    
    per_hour_mae = np.mean(np.abs(targets_unscaled - preds_unscaled), axis=0).flatten()
    
    if verbose:
        print("Validatie metrics (car count):")
        print(f"- MAE : {mae:.3f}")
        print(f"- RMSE: {rmse:.3f}")
        print(f"- MAPE: {mape:.2f}%")
        print("Gemiddelde MAE per voorspelde uur (0-23):")
        print(", ".join(f"{v:.2f}" for v in per_hour_mae))
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "per_hour_mae": per_hour_mae.tolist()
    }


if __name__ == "__main__":
    evaluate()
