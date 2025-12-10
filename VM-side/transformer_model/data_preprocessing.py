"""
Data preprocessing voor het transformer traffic prediction model.
Laadt data uit de database en bereidt deze voor op training.
Inclusief temporal features: uur van de dag, dag van de week, weekend indicator.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

from .config import DATABASE_URL, MODEL_CONFIG, TRAIN_CONFIG, MODEL_PATH


class TrafficDataset(Dataset):
    """PyTorch Dataset voor traffic data met temporal features."""
    
    def __init__(self, data: np.ndarray, seq_len: int = 24, pred_len: int = 24):
        """
        Args:
            data: Traffic data met features (N, 4) - [car_normalized, hour_normalized, day_normalized, is_weekend]
            seq_len: Input sequence length
            pred_len: Prediction length
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        # Target is alleen de car counts (eerste kolom)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, 0:1]
        return x, y


def load_data_from_db() -> pd.DataFrame:
    """Laad traffic data uit de database."""
    engine = create_engine(DATABASE_URL)
    query = """
        SELECT ts, car, camera_id 
        FROM traffic_samples 
        ORDER BY ts ASC
    """
    df = pd.read_sql(query, engine)
    return df


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregeer data per uur en voeg temporal features toe."""
    if df.empty:
        return pd.DataFrame(columns=['hour', 'car', 'hour_of_day', 'day_of_week', 'is_weekend'])
    
    df['ts'] = pd.to_datetime(df['ts'])
    df['hour'] = df['ts'].dt.floor('h')
    
    hourly = df.groupby('hour').agg({
        'car': 'sum'
    }).reset_index()
    
    # Voeg temporal features toe
    hourly['hour_of_day'] = hourly['hour'].dt.hour
    hourly['day_of_week'] = hourly['hour'].dt.dayofweek
    hourly['is_weekend'] = (hourly['day_of_week'] >= 5).astype(int)
    
    return hourly


def fill_missing_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Vul ontbrekende uren op met interpolatie."""
    if df.empty or len(df) < 2:
        return df
    
    df = df.set_index('hour')
    
    # Maak complete hourly range
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='h'
    )
    
    df = df.reindex(full_range)
    
    # Interpoleer car counts
    df['car'] = df['car'].interpolate(method='linear').fillna(0)
    
    # Vul temporal features opnieuw in (deze kunnen we exact berekenen)
    df = df.reset_index()
    df.columns = ['hour', 'car', 'hour_of_day', 'day_of_week', 'is_weekend']
    
    # Herbereken temporal features voor alle rijen
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def add_temporal_features(data: np.ndarray, timestamps: pd.Series) -> np.ndarray:
    """
    Voeg temporal features toe aan de data.
    
    Args:
        data: Car counts (N, 1)
        timestamps: Pandas series met timestamps
        
    Returns:
        Data met features (N, 4) - [car, hour_normalized, day_normalized, is_weekend]
    """
    hours = timestamps.dt.hour.values / 23.0  # Normaliseer 0-23 naar 0-1
    days = timestamps.dt.dayofweek.values / 6.0  # Normaliseer 0-6 naar 0-1
    is_weekend = (timestamps.dt.dayofweek >= 5).astype(float).values
    
    features = np.column_stack([data, hours, days, is_weekend])
    return features


def prepare_data(scaler_path: str = None):
    """
    Bereid data voor op training met temporal features.
    
    Returns:
        train_loader: DataLoader voor training
        val_loader: DataLoader voor validatie (kan None zijn als te weinig data)
        scaler: Fitted MinMaxScaler (alleen voor car counts)
    """
    # Laad en verwerk data
    df = load_data_from_db()
    
    if df.empty:
        raise ValueError("Geen data gevonden in de database. Voer eerst het data generatie script uit.")
    
    hourly = aggregate_hourly(df)
    hourly = fill_missing_hours(hourly)
    
    print(f"Total hourly data points: {len(hourly)}")
    
    # Normaliseer alleen car counts met scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    car_normalized = scaler.fit_transform(hourly[['car']].values)
    
    # Voeg temporal features toe (al genormaliseerd)
    hours_normalized = hourly['hour_of_day'].values.reshape(-1, 1) / 23.0
    days_normalized = hourly['day_of_week'].values.reshape(-1, 1) / 6.0
    is_weekend = hourly['is_weekend'].values.reshape(-1, 1).astype(float)
    
    # Combineer alle features: [car, hour, day, is_weekend]
    data = np.hstack([car_normalized, hours_normalized, days_normalized, is_weekend])
    
    print(f"Feature shape: {data.shape}")  # Should be (N, 4)
    
    # Sla scaler op
    os.makedirs(MODEL_PATH, exist_ok=True)
    scaler_file = scaler_path or os.path.join(MODEL_PATH, "scaler.pkl")
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Split data
    seq_len = MODEL_CONFIG['seq_len']
    pred_len = MODEL_CONFIG['pred_len']
    
    # Minimum data nodig: seq_len + pred_len
    min_data_needed = seq_len + pred_len
    
    if len(data) < min_data_needed:
        raise ValueError(f"Te weinig data: {len(data)} punten, minimaal {min_data_needed} nodig")
    
    # Gebruik 80% voor training, maar zorg dat er genoeg is voor beide
    split_idx = int(len(data) * TRAIN_CONFIG['train_split'])
    
    # Zorg dat train en val minstens 1 sequence kunnen maken
    min_for_sequence = seq_len + pred_len
    if split_idx < min_for_sequence:
        split_idx = min_for_sequence
    if len(data) - split_idx < min_for_sequence:
        # Niet genoeg voor validatie, gebruik alles voor training
        split_idx = len(data)
    
    train_data = data[:split_idx]
    val_data = data[split_idx:] if split_idx < len(data) else None
    
    # Maak datasets
    train_dataset = TrafficDataset(train_data, seq_len, pred_len)
    
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Maak dataloaders
    batch_size = min(TRAIN_CONFIG['batch_size'], len(train_dataset))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = None
    if val_data is not None and len(val_data) >= min_for_sequence:
        val_dataset = TrafficDataset(val_data, seq_len, pred_len)
        if len(val_dataset) > 0:
            val_batch_size = min(TRAIN_CONFIG['batch_size'], len(val_dataset))
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                drop_last=False
            )
            print(f"Validation dataset size: {len(val_dataset)}")
    
    if val_loader is None:
        print("No validation set (not enough data)")
    
    return train_loader, val_loader, scaler


def get_last_24_hours() -> tuple:
    """
    Haal de laatste 24 uur aan data op voor prediction.
    
    Returns:
        tuple: (car_counts (24, 1), timestamps (24,))
    """
    df = load_data_from_db()
    
    if df.empty:
        # Return dummy data als er geen data is
        now = datetime.now()
        dummy_timestamps = pd.date_range(end=now, periods=24, freq='h')
        return np.zeros((24, 1)), dummy_timestamps
    
    hourly = aggregate_hourly(df)
    hourly = fill_missing_hours(hourly)
    
    # Neem laatste 24 uur
    if len(hourly) >= 24:
        last_24 = hourly.tail(24)
        car_counts = last_24['car'].values.reshape(-1, 1)
        timestamps = last_24['hour']
    else:
        # Pad met nullen als we minder dan 24 uur hebben
        now = datetime.now()
        dummy_timestamps = pd.date_range(end=now, periods=24, freq='h')
        car_counts = np.zeros((24, 1))
        if len(hourly) > 0:
            car_counts[-len(hourly):, 0] = hourly['car'].values
        timestamps = dummy_timestamps
    
    return car_counts, timestamps


def create_features_for_prediction(car_counts: np.ndarray, timestamps: pd.DatetimeIndex, scaler) -> np.ndarray:
    """
    Maak features voor prediction input.
    
    Args:
        car_counts: Car counts (24, 1)
        timestamps: Timestamps voor de 24 uur
        scaler: Fitted scaler voor car counts
        
    Returns:
        Features array (24, 4) - [car_normalized, hour_normalized, day_normalized, is_weekend]
    """
    # Normaliseer car counts
    car_normalized = scaler.transform(car_counts)
    
    # Temporal features (al genormaliseerd)
    if isinstance(timestamps, pd.DatetimeIndex):
        hours = timestamps.hour.values
        days = timestamps.dayofweek.values
    else:
        hours = np.array([t.hour for t in timestamps])
        days = np.array([t.dayofweek() if hasattr(t, 'dayofweek') else t.weekday() for t in timestamps])
    
    hours_normalized = hours.reshape(-1, 1) / 23.0
    days_normalized = days.reshape(-1, 1) / 6.0
    is_weekend = (days >= 5).reshape(-1, 1).astype(float)
    
    # Combineer
    features = np.hstack([car_normalized, hours_normalized, days_normalized, is_weekend])
    return features


def create_features_for_date(target_date: datetime, scaler, last_car_counts: np.ndarray = None) -> np.ndarray:
    """
    Maak features voor een specifieke target datum.
    
    Args:
        target_date: De datum waarvoor features gemaakt worden
        scaler: Fitted scaler voor car counts
        last_car_counts: Optionele laatste car counts als basis (24, 1)
        
    Returns:
        Features array (24, 4)
    """
    # Genereer timestamps voor de target date
    target_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps = pd.date_range(start=target_start, periods=24, freq='h')
    
    # Als we geen historische data hebben, gebruik gemiddelde
    if last_car_counts is None:
        # Dummy waarden
        last_car_counts = np.ones((24, 1)) * 20  # Gemiddelde schatting
    
    return create_features_for_prediction(last_car_counts, timestamps, scaler)


def load_scaler():
    """Laad de opgeslagen scaler."""
    scaler_file = os.path.join(MODEL_PATH, "scaler.pkl")
    
    if not os.path.exists(scaler_file):
        return None
    
    with open(scaler_file, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Test data loading
    print("Loading data from database...")
    df = load_data_from_db()
    print(f"Loaded {len(df)} records")
    
    if not df.empty:
        hourly = aggregate_hourly(df)
        print(f"Hourly aggregated: {len(hourly)} records")
        
        hourly = fill_missing_hours(hourly)
        print(f"After filling gaps: {len(hourly)} records")
        print(hourly.head(10))
