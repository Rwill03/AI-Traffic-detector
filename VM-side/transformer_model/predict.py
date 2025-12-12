"""
Prediction script voor het transformer traffic prediction model.
Laadt een getraind model en maakt voorspellingen met temporal features.
"""
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from .config import BEST_MODEL_PATH, MODEL_PATH, get_device
from .model import create_model
from .data_preprocessing import get_last_24_hours, load_scaler, create_features_for_prediction, create_features_for_date


class TrafficPredictor:
    """Class voor het maken van traffic predictions met temporal features."""
    
    def __init__(self):
        self.device = get_device()
        self.model = None
        self.scaler = None
        self._load_model()
        
    def _load_model(self):
        """Laad het getrainde model."""
        if not os.path.exists(BEST_MODEL_PATH):
            print(f"Warning: No trained model found at {BEST_MODEL_PATH}")
            return False
            
        self.model = create_model(self.device)
        
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.scaler = load_scaler()
        
        print(f"Model loaded from epoch {checkpoint['epoch']+1}")
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")
        
        return True
    
    def is_ready(self) -> bool:
        """Check of het model klaar is voor predictions."""
        return self.model is not None and self.scaler is not None
    
    def predict_next_24_hours(self, base_time: datetime = None) -> List[Dict]:
        """
        Voorspel het aantal auto's voor de volgende 24 uur.
        
        Args:
            base_time: Starttijd voor de voorspelling (default: nu)
            
        Returns:
            List van dicts met 'hour' en 'predicted_cars'
        """
        if not self.is_ready():
            return self._generate_dummy_predictions(base_time)
        
        if base_time is None:
            base_time = datetime.now()
        
        # Haal laatste 24 uur data op
        car_counts, timestamps = get_last_24_hours()
        
        # Maak features met temporal info
        features = create_features_for_prediction(car_counts, timestamps, self.scaler)
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Maak voorspelling
        with torch.no_grad():
            output = self.model.predict(input_tensor)
        
        # Denormaliseer output (alleen car counts)
        output_np = output.cpu().numpy().reshape(-1, 1)
        predictions = self.scaler.inverse_transform(output_np).flatten()
        
        # Zorg dat predictions niet negatief zijn
        predictions = np.maximum(predictions, 0)
        
        # Maak resultaat
        results = []
        for i, pred in enumerate(predictions):
            hour_time = base_time + timedelta(hours=i)
            results.append({
                'hour': hour_time.strftime('%Y-%m-%d %H:00'),
                'hour_of_day': hour_time.hour,
                'predicted_cars': int(round(pred)),
                'is_rush_hour': hour_time.hour in [7, 8, 9, 16, 17, 18]
            })
        
        return results
    
    def predict_for_date(self, target_date: datetime) -> List[Dict]:
        """
        Voorspel het aantal auto's voor een specifieke datum.
        Houdt rekening met dag van de week (weekend vs doordeweeks).
        
        Args:
            target_date: De datum waarvoor voorspeld moet worden
            
        Returns:
            List van dicts met hourly predictions
        """
        # Zet target_date naar begin van de dag
        target_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if not self.is_ready():
            return self._generate_dummy_predictions(target_start)
        
        # Haal laatste bekende car counts op als basis
        car_counts, _ = get_last_24_hours()
        
        # Maak features voor de target datum (met juiste dag/weekend info)
        features = create_features_for_date(target_start, self.scaler, car_counts)
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Maak voorspelling
        with torch.no_grad():
            output = self.model.predict(input_tensor)
        
        # Denormaliseer output
        output_np = output.cpu().numpy().reshape(-1, 1)
        predictions = self.scaler.inverse_transform(output_np).flatten()
        predictions = np.maximum(predictions, 0)
        
        # Maak resultaat
        results = []
        for i, pred in enumerate(predictions):
            hour_time = target_start + timedelta(hours=i)
            results.append({
                'hour': hour_time.strftime('%Y-%m-%d %H:00'),
                'hour_of_day': hour_time.hour,
                'predicted_cars': int(round(pred)),
                'is_rush_hour': hour_time.hour in [7, 8, 9, 16, 17, 18]
            })
        
        return results
    
    def _generate_dummy_predictions(self, base_time: datetime = None) -> List[Dict]:
        """Genereer dummy predictions als het model niet beschikbaar is."""
        if base_time is None:
            base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        results = []
        for i in range(24):
            hour_time = base_time + timedelta(hours=i)
            hour = hour_time.hour
            
            # Simuleer spitsuur patronen
            if 7 <= hour <= 9 or 16 <= hour <= 18:
                predicted = np.random.randint(40, 70)
            elif 10 <= hour <= 15:
                predicted = np.random.randint(20, 40)
            elif 6 <= hour <= 22:
                predicted = np.random.randint(10, 25)
            else:
                predicted = np.random.randint(2, 10)
            
            results.append({
                'hour': hour_time.strftime('%Y-%m-%d %H:00'),
                'hour_of_day': hour,
                'predicted_cars': predicted,
                'is_rush_hour': hour in [7, 8, 9, 16, 17, 18],
                'is_dummy': True
            })
        
        return results
    
    def get_current_hour_prediction(self) -> Dict:
        """Haal de voorspelling voor het huidige uur op."""
        predictions = self.predict_next_24_hours()
        return predictions[0] if predictions else None
    
    def get_daily_summary(self, predictions: List[Dict]) -> Dict:
        """
        Maak een samenvatting van de dagvoorspellingen.
        
        Args:
            predictions: List van hourly predictions
            
        Returns:
            Dict met summary statistics
        """
        if not predictions:
            return {}
        
        cars = [p['predicted_cars'] for p in predictions]
        rush_hours = [p for p in predictions if p.get('is_rush_hour', False)]
        
        return {
            'total_predicted': sum(cars),
            'average_per_hour': round(np.mean(cars), 1),
            'peak_hour': predictions[np.argmax(cars)]['hour'],
            'peak_cars': max(cars),
            'quietest_hour': predictions[np.argmin(cars)]['hour'],
            'min_cars': min(cars),
            'rush_hour_total': sum(p['predicted_cars'] for p in rush_hours),
            'rush_hour_average': round(np.mean([p['predicted_cars'] for p in rush_hours]), 1) if rush_hours else 0
        }


# Global predictor instance
_predictor: Optional[TrafficPredictor] = None


def get_predictor() -> TrafficPredictor:
    """Haal de global predictor instance op (lazy loading)."""
    global _predictor
    if _predictor is None:
        _predictor = TrafficPredictor()
    return _predictor


def predict_next_24_hours(base_time: datetime = None) -> List[Dict]:
    """Convenience function voor predictions."""
    return get_predictor().predict_next_24_hours(base_time)


def predict_for_date(target_date: datetime) -> List[Dict]:
    """Convenience function voor date-based predictions."""
    return get_predictor().predict_for_date(target_date)


if __name__ == "__main__":
    from datetime import date
    
    print("Testing Traffic Predictor...")
    print("-" * 50)
    
    predictor = TrafficPredictor()
    
    if predictor.is_ready():
        print("Model loaded successfully!")
    else:
        print("Using dummy predictions (no trained model found)")
    
    print("\nPredictions for next 24 hours:")
    predictions = predictor.predict_next_24_hours()
    
    for pred in predictions:
        rush = "🚗" if pred.get('is_rush_hour') else "  "
        print(f"  {rush} {pred['hour']}: {pred['predicted_cars']} cars")
    
    print("\nDaily Summary:")
    summary = predictor.get_daily_summary(predictions)
    for key, value in summary.items():
        print(f"  {key}: {value}")
