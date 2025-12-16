"""
Training script voor het transformer traffic prediction model.
Ondersteunt GPU training met NVIDIA CUDA.
"""
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime
import json

from .config import TRAIN_CONFIG, MODEL_PATH, BEST_MODEL_PATH, get_device
from .model import create_model, count_parameters
from .data_preprocessing import prepare_data


class EarlyStopping:
    """Early stopping om overfitting te voorkomen."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train één epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (encoder-only, geen teacher forcing nodig)
        output = model(src)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Valideer het model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            output = model(src)  # Geen teacher forcing bij validatie
            loss = criterion(output, tgt)
            
            total_loss += loss.item()
            
    return total_loss / len(val_loader)


def train_model(epochs: int = None, verbose: bool = True):
    """
    Train het transformer model.
    
    Args:
        epochs: Aantal epochs (default uit config)
        verbose: Print training progress
        
    Returns:
        dict met training history
    """
    if epochs is None:
        epochs = TRAIN_CONFIG['epochs']
        
    # Setup
    device = get_device()
    
    if verbose:
        print(f"Training on: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Data
    if verbose:
        print("\nLoading and preparing data...")
    
    try:
        train_loader, val_loader, scaler = prepare_data()
    except ValueError as e:
        print(f"Error: {e}")
        return None
    
    if verbose:
        print(f"Training batches: {len(train_loader)}")
        if val_loader:
            print(f"Validation batches: {len(val_loader)}")
        else:
            print("No validation set available")
    
    # Model
    model = create_model(device)
    
    if verbose:
        print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Training setup
    # Huber (Smooth L1) is wat robuuster tegen uitschieters dan MSE
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=TRAIN_CONFIG['early_stopping_patience'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    # Maak model directory
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    if verbose:
        print("\nStarting training...")
        print("-" * 50)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validatie alleen als we een val_loader hebben
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
        else:
            val_loss = train_loss  # Gebruik train loss als proxy
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, BEST_MODEL_PATH)
            
            if verbose:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | ★ Best")
        else:
            if verbose:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save training history
    history['training_completed'] = datetime.now().isoformat()
    history_path = os.path.join(MODEL_PATH, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    if verbose:
        print("-" * 50)
        print(f"Training completed!")
        print(f"Best validation loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch']+1}")
        print(f"Model saved to: {BEST_MODEL_PATH}")
    
    return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train traffic prediction model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    history = train_model(epochs=args.epochs, verbose=not args.quiet)
    
    if history:
        print(f"\nFinal training loss: {history['train_loss'][-1]:.6f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
