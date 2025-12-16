"""
Configuratie voor het transformer traffic prediction model.
"""
import os
import torch

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://traffic_user:supersecretpassword@db/traffic_db"
)

# Model hyperparameters
MODEL_CONFIG = {
    "input_dim": 4,           # aantal input features (car, hour, day_of_week, is_weekend)
    "output_dim": 1,          # aantal output features (alleen car count)
    "d_model": 64,            # transformer dimensie
    "nhead": 4,               # aantal attention heads
    "num_encoder_layers": 2,  # aantal encoder layers
    "dim_feedforward": 128,   # feedforward dimensie
    "dropout": 0.1,           # dropout rate
    "seq_len": 24,            # input sequence length (24 uur)
    "pred_len": 24,           # prediction length (24 uur)
}

# Training parameters
TRAIN_CONFIG = {
    "batch_size": 4,
    "epochs": 150,
    "learning_rate": 0.0005,  # Lagere learning rate voor betere convergentie
    "train_split": 0.8,
    "early_stopping_patience": 15,  # Meer patience voor betere training
}

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models")
BEST_MODEL_PATH = os.path.join(MODEL_PATH, "best_model.pth")

def get_device():
    """Return het beste beschikbare device (GPU of CPU)."""
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")
