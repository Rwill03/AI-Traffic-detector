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
        Gebruikt historische gemiddelden per dag/uur combinatie als basis.
        
        Args:
            target_date: De datum waarvoor voorspeld moet worden
            
        Returns:
            List van dicts met hourly predictions
        """
        # Zet target_date naar begin van de dag
        target_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if not self.is_ready():
            return self._generate_dummy_predictions(target_start)
        
        # Bereken dag van week en weekend voor target date
        target_day_of_week = target_start.weekday()
        is_weekend = target_day_of_week >= 5
        
        # Haal historische gemiddelden op voor deze dag/uur combinatie
        car_counts = self._get_historical_pattern(target_day_of_week)
        
        # Als het weekend is, gebruik het historische profiel direct (stabieler dan model-output op weekday data)
        if is_weekend and car_counts is not None:
            results = []
            for i, val in enumerate(car_counts.flatten()):
                hour_time = target_start + timedelta(hours=i)
                results.append({
                    'hour': hour_time.strftime('%Y-%m-%d %H:00'),
                    'hour_of_day': hour_time.hour,
                    'predicted_cars': int(round(val)),
                    'is_rush_hour': hour_time.hour in [7, 8, 9, 16, 17, 18]
                })
            return results
        
        # Maak features voor de target datum
        features = create_features_for_date(target_start, self.scaler, car_counts)
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Maak voorspelling
        with torch.no_grad():
            output = self.model.predict(input_tensor)
        
        # Denormaliseer output
        output_np = output.cpu().numpy().reshape(-1, 1)
        predictions = self.scaler.inverse_transform(output_np).flatten()
        predictions = np.maximum(predictions, 0)
        
        # Post-processing: pas dag-specifieke correctie toe gebaseerd op historische data
        predictions = self._apply_daytype_correction(predictions, target_day_of_week)
        
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
    
    def _apply_daytype_correction(self, predictions: np.ndarray, target_day_of_week: int) -> np.ndarray:
        """
        Pas dag-specifieke correctie toe op basis van historische patronen.
        Herschaalt predictions naar het historisch gemiddelde voor die specifieke dag.
        
        Args:
            predictions: Raw model predictions
            target_day_of_week: 0=maandag, 6=zondag
            
        Returns:
            Gecorrigeerde predictions
        """
        try:
            from .data_preprocessing import load_data_from_db, aggregate_hourly
            
            df = load_data_from_db()
            if df.empty:
                # Geen correctie mogelijk
                is_weekend = target_day_of_week >= 5
                return predictions * (0.4 if is_weekend else 1.0)
            
            hourly = aggregate_hourly(df)
            hourly['hour_of_day'] = hourly['hour'].dt.hour
            hourly['day_of_week'] = hourly['hour'].dt.dayofweek
            
            # Bereken gemiddelde per uur voor ALLE weekdagen (baseline model)
            all_weekdays = hourly[hourly['day_of_week'] < 5].groupby('hour_of_day')['car'].mean()
            
            # Bereken gemiddelde per uur voor de TARGET dag
            target_day = hourly[hourly['day_of_week'] == target_day_of_week].groupby('hour_of_day')['car'].mean()
            
            # Als target_day onvoldoende data heeft, gebruik weekend/weekday gemiddelde
            if len(target_day) < 12:
                is_weekend = target_day_of_week >= 5
                if is_weekend:
                    target_day = hourly[hourly['day_of_week'] >= 5].groupby('hour_of_day')['car'].mean()
                else:
                    target_day = all_weekdays
            
            corrected = predictions.copy()
            for hour in range(24):
                if hour in all_weekdays.index and hour in target_day.index:
                    if all_weekdays[hour] > 0:
                        # Bereken ratio van target dag t.o.v. weekday baseline
                        ratio = target_day[hour] / all_weekdays[hour]
                        # Clamp ratio om extreme afwijkingen door uitschieters te vermijden
                        ratio = max(0.5, min(1.5, ratio))
                        corrected[hour] *= ratio
                elif hour in target_day.index:
                    # Gebruik absolute waarde uit historische data
                    corrected[hour] = target_day[hour]
            
            return corrected
            
        except Exception as e:
            print(f"Error applying daytype correction: {e}")
            # Simpele fallback correctie
            is_weekend = target_day_of_week >= 5
            return predictions * (0.4 if is_weekend else 1.0)
    
    def _get_historical_pattern(self, target_day_of_week: int) -> np.ndarray:
        """
        Haal historisch gemiddelde patroon op voor een specifieke dag (weekday/weekend bewust).
        
        Args:
            target_day_of_week: 0=maandag, 6=zondag
            
        Returns:
            Array van 24 car counts
        """
        try:
            from .data_preprocessing import load_data_from_db, aggregate_hourly
            
            df = load_data_from_db()
            if df.empty:
                return self._generate_dummy_pattern(target_day_of_week)
            
            hourly = aggregate_hourly(df)
            hourly['hour_of_day'] = hourly['hour'].dt.hour
            hourly['day_of_week'] = hourly['hour'].dt.dayofweek
            
            # Prefer exact day-of-week gemiddelde; val terug naar weekend/weekday gemiddelde
            target = hourly[hourly['day_of_week'] == target_day_of_week].groupby('hour_of_day')['car'].mean()
            if len(target) < 12:
                if target_day_of_week >= 5:
                    target = hourly[hourly['day_of_week'] >= 5].groupby('hour_of_day')['car'].mean()
                else:
                    target = hourly[hourly['day_of_week'] < 5].groupby('hour_of_day')['car'].mean()
            
            if len(target) == 0:
                target = hourly.groupby('hour_of_day')['car'].mean()
            
            fallback_val = target.mean() if len(target) > 0 else 20
            result = np.zeros(24)
            for hour in range(24):
                result[hour] = target.get(hour, fallback_val)
            
            return result.reshape(-1, 1)
            
        except Exception as e:
            print(f"Error getting historical pattern: {e}")
            return np.ones((24, 1)) * 20
    
    def _generate_dummy_pattern(self, day_of_week: int) -> np.ndarray:
        """Genereer een dummy patroon op basis van dag van de week."""
        is_weekend = day_of_week >= 5
        pattern = np.zeros(24)
        
        for hour in range(24):
            if is_weekend:
                # Weekend: lager verkeer, geen echte spits
                if 7 <= hour <= 9 or 16 <= hour <= 18:
                    pattern[hour] = 30 + np.random.randint(-5, 5)
                elif 10 <= hour <= 15:
                    pattern[hour] = 25 + np.random.randint(-5, 5)
                elif 6 <= hour <= 22:
                    pattern[hour] = 15 + np.random.randint(-5, 5)
                else:
                    pattern[hour] = 3 + np.random.randint(-2, 2)
            else:
                # Weekday: duidelijke spitsuren
                if 7 <= hour <= 9 or 16 <= hour <= 18:
                    pattern[hour] = 60 + np.random.randint(-10, 10)
                elif 10 <= hour <= 15:
                    pattern[hour] = 30 + np.random.randint(-5, 5)
                elif 6 <= hour <= 22:
                    pattern[hour] = 20 + np.random.randint(-5, 5)
                else:
                    pattern[hour] = 3 + np.random.randint(-2, 2)
        
        return pattern.reshape(-1, 1)
    
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
