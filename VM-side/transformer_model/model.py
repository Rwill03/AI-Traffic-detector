"""
Transformer model architectuur voor traffic prediction.
Gebruikt een encoder-only approach met direct output projection.
"""
import torch
import torch.nn as nn
import math

from .config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """Positional encoding voor de transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrafficTransformer(nn.Module):
    """
    Transformer encoder model voor traffic prediction.
    Voorspelt het aantal auto's per uur voor de volgende 24 uur.
    
    Input features: [car_normalized, hour_normalized, day_normalized, is_weekend]
    Output: [predicted_car_normalized] voor elk uur
    """
    
    def __init__(
        self,
        input_dim: int = MODEL_CONFIG['input_dim'],
        output_dim: int = MODEL_CONFIG.get('output_dim', 1),
        d_model: int = MODEL_CONFIG['d_model'],
        nhead: int = MODEL_CONFIG['nhead'],
        num_encoder_layers: int = MODEL_CONFIG['num_encoder_layers'],
        num_decoder_layers: int = MODEL_CONFIG['num_decoder_layers'],
        dim_feedforward: int = MODEL_CONFIG['dim_feedforward'],
        dropout: float = MODEL_CONFIG['dropout'],
        seq_len: int = MODEL_CONFIG['seq_len'],
        pred_len: int = MODEL_CONFIG['pred_len']
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # Input embedding (4 features -> d_model)
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 100, dropout=dropout)
        
        # Transformer encoder only
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output projection: van encoded sequence naar predictions
        # Flatten de encoded output en project naar pred_len outputs
        self.flatten = nn.Flatten()
        self.output_projection = nn.Sequential(
            nn.Linear(seq_len * d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_len * output_dim),
            nn.Sigmoid()  # Output tussen 0 en 1 (genormaliseerde car counts)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialiseer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, tgt=None):
        """
        Forward pass.
        
        Args:
            src: Source sequence (batch, seq_len, input_dim)
            tgt: Target sequence (niet gebruikt, voor backwards compatibility)
            
        Returns:
            Output predictions (batch, pred_len, input_dim)
        """
        batch_size = src.size(0)
        
        # Embed input
        x = self.input_embedding(src)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Flatten en project naar output
        x = self.flatten(x)  # (batch, seq_len * d_model)
        x = self.output_projection(x)  # (batch, pred_len * output_dim)
        
        # Reshape naar (batch, pred_len, output_dim)
        output = x.view(batch_size, self.pred_len, self.output_dim)
        
        return output
    
    def predict(self, src):
        """
        Voorspel de volgende 24 uur.
        
        Args:
            src: Source sequence (batch, seq_len, input_dim) of (seq_len, input_dim)
            
        Returns:
            Predictions (batch, pred_len, input_dim) of (pred_len, input_dim)
        """
        was_2d = src.dim() == 2
        if was_2d:
            src = src.unsqueeze(0)
            
        self.eval()
        with torch.no_grad():
            output = self.forward(src)
            
        if was_2d:
            output = output.squeeze(0)
            
        return output


def create_model(device=None):
    """Maak een nieuw model aan."""
    model = TrafficTransformer()
    
    if device is not None:
        model = model.to(device)
        
    return model


def count_parameters(model):
    """Tel het aantal trainbare parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    from .config import get_device
    
    device = get_device()
    model = create_model(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = MODEL_CONFIG['seq_len']
    pred_len = MODEL_CONFIG['pred_len']
    
    src = torch.randn(batch_size, seq_len, 1).to(device)
    tgt = torch.randn(batch_size, pred_len, 1).to(device)
    
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, pred_len, 1)
    
    # Test inference
    pred = model.predict(src)
    print(f"Prediction shape: {pred.shape}")
