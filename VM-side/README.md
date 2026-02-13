# Traffic Prediction API

A FastAPI server with a Transformer-based machine learning model for predicting traffic intensity per hour.

## 🚀 Quick Start

### Requirements
- Docker & Docker Compose
- NVIDIA GPU with CUDA support (optional, but recommended)
- NVIDIA Container Toolkit (for GPU support)

### 1. Starting the project

```bash
# Start the containers (API + PostgreSQL database)
docker compose up -d

# Check if everything is running
docker compose ps
```

The API is now available at `http://localhost:8001`

### 2. Opening the dashboard

Open your browser and go to:
```
http://localhost:8001/
```

You'll see two tabs:
- **Status**: Live traffic observations
- **Predictions**: AI predictions per hour

---

## 🤖 Model Training

### Step 1: Filling the database with data

If the database is empty, generate test data first:

```bash
# Generate 7 days of realistic traffic data
python3 debug/generate_realistic_data.py --days 7
```

Or add real observations via the API:
```bash
curl -X POST http://localhost:8001/api/v1/observation \
  -H "Content-Type: application/json" \
  -d '{"ts": "2025-12-10T08:30:00", "car": 45, "truck": 3, "bus": 1}'
```

### Step 2: Training the model

```bash
# Train the model (in the Docker container)
docker compose exec api python3 -c "
from transformer_model.train import train_model
train_model(epochs=100, verbose=True)
"
```

### Validation & evaluation
- Training now always uses a hold-out validation set (minimum 24 hours input + 24 hours target, so at least 96 hourly points needed).
- Run validation metrics (MAE/RMSE/MAPE) on the saved `best_model.pth`:
```bash
docker compose exec api python3 "Evaluation model/evaluate_model.py"
```
- How it works:
  - Data split: last 20% of hourly data is validation, with at least one full sequence of 24h input + 24h target for both train and val.
  - Metrics: MAE (mean absolute error) and RMSE (root mean squared error) in real car counts; sMAPE as percentage (more stable with low counts); per-hour MAE for insight into specific hours.
  - Training: loss is SmoothL1/Huber (mix of MAE/MSE; dampens outliers), optimizer is Adam (adaptive lr) with weight decay; best val-loss saves `best_model.pth` via early stopping + LR scheduler.
  - Pipelines: scaler is reused, model is loaded from `best_model.pth` (best val-loss during training), validation set is passed through the model, then metrics are calculated and shown in the dashboard via `/api/v1/model/metrics`.

Training parameters can be adjusted in `transformer_model/config.py`:
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size (default: 4)
- `learning_rate`: Learning rate (default: 0.001)
- `early_stopping_patience`: Stop after N epochs without improvement (default: 10)

### Step 3: Restarting the API to load the new model

```bash
docker compose restart api
```

---

## 📁 Project Structure

```
VM-side/
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── static/
│   └── dashboard.html         # Web dashboard
└── transformer_model/
    ├── __init__.py
    ├── config.py              # Model configuration
    ├── model.py               # Transformer architecture
    ├── data_preprocessing.py  # Data loading & features
    ├── train.py               # Training script
    ├── predict.py             # Prediction logic
    └── saved_models/          # Saved models
        ├── best_model.pth     # Best model weights
        └── scaler.pkl         # Data scaler
```

---

## 🔧 API Endpoints

### Status & Observations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML |
| `/api/v1/status` | GET | Server status + latest observations |
| `/api/v1/observation` | POST | Add new observation |

### Predictions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predictions?date=YYYY-MM-DD` | GET | 24-hour prediction for date |
| `/api/v1/predictions/current` | GET | Prediction for current hour |
| `/api/v1/predictions/week` | GET | Predictions for entire week |

### Example Response
```json
{
  "predictions": [
    {"hour": "2025-12-10 07:00", "hour_of_day": 7, "predicted_cars": 62, "is_rush_hour": true},
    {"hour": "2025-12-10 08:00", "hour_of_day": 8, "predicted_cars": 68, "is_rush_hour": true}
  ],
  "summary": {
    "total_predicted": 581,
    "peak_hour": "2025-12-10 08:00",
    "peak_cars": 68,
    "rush_hour_average": 56
  },
  "model_ready": true
}
```

---

## 🧠 Model Details

### Architecture
- **Type**: Transformer Encoder
- **Input features**: 4 (car_count, hour, day_of_week, is_weekend)
- **Output**: 24 hour predictions
- **Parameters**: ~267,000

### Features
The model uses temporal features to learn different patterns:
- **Hour of day**: Rush hour vs night
- **Day of week**: Monday through Sunday
- **Weekend indicator**: Weekends have less traffic

### Typical predictions
| Day Type | Rush Hour (7-9 AM) | Night (0-5 AM) |
|----------|-------------------|----------------|
| Weekday | 60-70 cars | 2-5 cars |
| Weekend | 30-40 cars | 2-5 cars |

---

## 🐳 Docker Commands

```bash
# Alles opstarten
docker compose up -d

# Logs bekijken
docker compose logs -f api

# In container komen
docker compose exec api bash

# Containers stoppen
docker compose down

# Volledig opnieuw bouwen
docker compose up -d --build

# GPU check
docker compose exec api python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## 🔄 Retraining after new data

If new observations have been added:

```bash
# 1. Retrain the model
docker compose exec api python3 -c "
from transformer_model.train import train_model
train_model(epochs=100, verbose=True)
"

# 2. Restart the API to load the new model
docker compose restart api

# 3. Verify that the model is loaded
curl http://localhost:8001/api/v1/predictions/current | jq '.model_ready'
```

---

## ⚠️ Troubleshooting

### Model only predicts 0
- Check if model is trained: `ls -la transformer_model/saved_models/`
- Retrain the model if `best_model.pth` is missing

### GPU is not being used
- Check NVIDIA drivers: `nvidia-smi`
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Database connection error
- Check if PostgreSQL is running: `docker compose ps`
- Check logs: `docker compose logs db`

### API doesn't start
- Check logs: `docker compose logs api`
- Rebuild: `docker compose up -d --build`
