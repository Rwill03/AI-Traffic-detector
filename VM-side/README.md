# Traffic Prediction API

Een FastAPI server met een Transformer-gebaseerd machine learning model voor het voorspellen van verkeersintensiteit per uur.

## 🚀 Quick Start

### Vereisten
- Docker & Docker Compose
- NVIDIA GPU met CUDA support (optioneel, maar aanbevolen)
- NVIDIA Container Toolkit (voor GPU support)

### 1. Project opstarten

```bash
# Start de containers (API + PostgreSQL database)
docker compose up -d

# Check of alles draait
docker compose ps
```

De API is nu beschikbaar op `http://localhost:8001`

### 2. Dashboard openen

Open je browser en ga naar:
```
http://localhost:8001/
```

Je ziet twee tabs:
- **Status**: Live verkeersobservaties
- **Predictions**: AI-voorspellingen per uur

---

## 🤖 Model Training

### Stap 1: Database vullen met data

Als de database leeg is, genereer eerst testdata:

```bash
# Genereer 7 dagen realistische verkeersdata
python3 debug/generate_realistic_data.py --days 7
```

Of voeg echte observaties toe via de API:
```bash
curl -X POST http://localhost:8001/api/v1/observation \
  -H "Content-Type: application/json" \
  -d '{"ts": "2025-12-10T08:30:00", "car": 45, "truck": 3, "bus": 1}'
```

### Stap 2: Model trainen

```bash
# Train het model (in de Docker container)
docker compose exec api python3 -c "
from transformer_model.train import train_model
train_model(epochs=100, verbose=True)
"
```

### Validatie & evaluatie
- Training gebruikt nu altijd een hold-out validatieset (minimaal 24 uur input + 24 uur target, dus minstens 96 uurlijkse punten nodig).
- Draai validatiemetrics (MAE/RMSE/MAPE) op de opgeslagen `best_model.pth`:
```bash
docker compose exec api python3 "Evaluation model/evaluate_model.py"
```
- Hoe het werkt:
  - Data split: laatste 20% uurdata is validatie, met minimaal één volledige sequence van 24u input + 24u target voor zowel train als val.
  - Metrics: MAE (gemiddelde absolute fout) en RMSE (wortel van kwadratische fout) in echte auto-aantallen; sMAPE als percentage (stabieler bij lage aantallen); per-uur MAE voor inzicht in specifieke uren.
  - Training: verlies is SmoothL1/Huber (mix van MAE/MSE; dempt uitschieters), optimizer is Adam (adaptieve lr) met weight decay; beste val-loss bewaart `best_model.pth` via early stopping + LR scheduler.
  - Pipelines: scaler wordt hergebruikt, model wordt geladen uit `best_model.pth` (beste val-loss tijdens training), validatieset wordt door het model gehaald, vervolgens worden metrics berekend en in het dashboard getoond via `/api/v1/model/metrics`.

Training parameters kunnen aangepast worden in `transformer_model/config.py`:
- `epochs`: Aantal training epochs (default: 100)
- `batch_size`: Batch grootte (default: 4)
- `learning_rate`: Learning rate (default: 0.001)
- `early_stopping_patience`: Stop na N epochs zonder verbetering (default: 10)

### Stap 3: API herstarten om nieuw model te laden

```bash
docker compose restart api
```

---

## 📁 Project Structuur

```
VM-side/
├── main.py                    # FastAPI applicatie
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuratie
├── static/
│   └── dashboard.html         # Web dashboard
└── transformer_model/
    ├── __init__.py
    ├── config.py              # Model configuratie
    ├── model.py               # Transformer architectuur
    ├── data_preprocessing.py  # Data loading & features
    ├── train.py               # Training script
    ├── predict.py             # Prediction logic
    └── saved_models/          # Opgeslagen models
        ├── best_model.pth     # Beste model weights
        └── scaler.pkl         # Data scaler
```

---

## 🔧 API Endpoints

### Status & Observaties
| Endpoint | Method | Beschrijving |
|----------|--------|--------------|
| `/` | GET | Dashboard HTML |
| `/api/v1/status` | GET | Server status + laatste observaties |
| `/api/v1/observation` | POST | Nieuwe observatie toevoegen |

### Predictions
| Endpoint | Method | Beschrijving |
|----------|--------|--------------|
| `/api/v1/predictions?date=YYYY-MM-DD` | GET | 24-uurs voorspelling voor datum |
| `/api/v1/predictions/current` | GET | Voorspelling voor huidig uur |
| `/api/v1/predictions/week` | GET | Voorspellingen voor hele week |

### Voorbeeld Response
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

### Architectuur
- **Type**: Transformer Encoder
- **Input features**: 4 (car_count, hour, day_of_week, is_weekend)
- **Output**: 24 uur voorspellingen
- **Parameters**: ~267,000

### Features
Het model gebruikt temporal features om verschillende patronen te leren:
- **Uur van de dag**: Rush hour vs nacht
- **Dag van de week**: Maandag t/m zondag
- **Weekend indicator**: Weekenden hebben minder verkeer

### Typische voorspellingen
| Dag Type | Rush Hour (7-9u) | Nacht (0-5u) |
|----------|------------------|--------------|
| Doordeweeks | 60-70 auto's | 2-5 auto's |
| Weekend | 30-40 auto's | 2-5 auto's |

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

## 🔄 Hertrainen na nieuwe data

Als er nieuwe observaties zijn toegevoegd:

```bash
# 1. Train het model opnieuw
docker compose exec api python3 -c "
from transformer_model.train import train_model
train_model(epochs=100, verbose=True)
"

# 2. Herstart de API om het nieuwe model te laden
docker compose restart api

# 3. Verifieer dat het model geladen is
curl http://localhost:8001/api/v1/predictions/current | jq '.model_ready'
```

---

## ⚠️ Troubleshooting

### Model geeft alleen 0 als voorspelling
- Controleer of het model getraind is: `ls -la transformer_model/saved_models/`
- Hertrain het model als `best_model.pth` ontbreekt

### GPU wordt niet gebruikt
- Check NVIDIA drivers: `nvidia-smi`
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Database connectie error
- Check of PostgreSQL draait: `docker compose ps`
- Check logs: `docker compose logs db`

### API start niet op
- Check logs: `docker compose logs api`
- Rebuild: `docker compose up -d --build`
