# 🚗 AI Traffic Detector

Een real-time verkeersbewakingssysteem dat AI-gestuurde voertuigdetectie combineert met een Transformer-model voor verkeersvoorspellingen.

![Architecture](https://img.shields.io/badge/Architecture-Edge%20%2B%20Cloud-blue)
![Model](https://img.shields.io/badge/Detection-RF--DETR-green)
![Prediction](https://img.shields.io/badge/Prediction-Transformer-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![CI/CD](https://github.com/Rwill03/AI-Traffic-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/Rwill03/AI-Traffic-detector/actions/workflows/ci.yml)

## 📋 Inhoudsopgave

- [Overzicht](#overzicht)
- [Architectuur](#architectuur)
- [Componenten](#componenten)
  - [Rock5 Edge Device](#rock5-edge-device-camera-detectie)
  - [VM Backend Server](#vm-backend-server)
  - [Transformer Prediction Model](#transformer-prediction-model)
- [Installatie & Opstarten](#installatie--opstarten)
- [API Endpoints](#api-endpoints)
- [Dashboard Interface](#dashboard-interface)
- [Model Evaluatie](#model-evaluatie)
- [CI/CD Pipeline](#cicd-pipeline)

---

## 🎯 Overzicht

Dit systeem detecteert en telt voertuigen in real-time via een camera aangesloten op een Rock5 edge device. De data wordt doorgestuurd naar een VM-server met GPU, waar een Transformer-model verkeerspatronen leert en voorspellingen maakt voor de komende 24 uur.

### Features

- ✅ **Real-time voertuigdetectie** met RF-DETR (COCO dataset)
- ✅ **Classificatie** per voertuigtype: auto, vrachtwagen, bus, motor, fiets
- ✅ **24-uurs voorspellingen** met AI Transformer model
- ✅ **Weekend/weekday bewustzijn** in voorspellingen
- ✅ **Spitsuur detectie** (7-9u en 16-18u)
- ✅ **Live dashboard** met monitoring en voorspellingen
- ✅ **GPU-acceleratie** voor snelle inference

---

## 🏗️ Architectuur

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ROCK5 EDGE DEVICE                             │
│  ┌──────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │  Camera  │───▶│  RF-DETR    │───▶│  POST API   │────────────────────┐│
│  │ /dev/video1   │  (CPU)      │    │  + Snapshot │                    ││
│  └──────────┘    └─────────────┘    └─────────────┘                    ││
│                   COCO Classes:                                         ││
│                   car, truck, bus,                                      ││
│                   motorcycle, bicycle                                   ││
└─────────────────────────────────────────────────────────────────────────┘│
                                                                           │
                              ▼  HTTP POST (elke 10s)                      │
                                                                           │
┌─────────────────────────────────────────────────────────────────────────┐│
│                           VM SERVER (GPU)                               ││
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ ││
│  │  FastAPI    │───▶│ PostgreSQL  │───▶│  Transformer Model          │ ││
│  │  :8001      │    │  Database   │    │  (PyTorch + CUDA)           │ ││
│  └─────────────┘    └─────────────┘    │  - 24h input sequence       │ ││
│        │                               │  - 24h prediction output    │ ││
│        ▼                               └─────────────────────────────┘ ││
│  ┌─────────────┐                                                        ││
│  │  Dashboard  │  ◀── Real-time updates elke 5s                        ││
│  │  (Web UI)   │                                                        ││
│  └─────────────┘                                                        ││
└─────────────────────────────────────────────────────────────────────────┘│
```

---

## 📦 Componenten

### Rock5 Edge Device (Camera Detectie)

**Locatie:** `camera_to_server/`

Het Rock5 board draait een headless Python script dat:
1. Frames leest van de USB camera (`/dev/video1`)
2. Elke 10 seconden inference uitvoert met RF-DETR
3. Voertuigen classificeert en telt
4. Snapshot + tellingen naar de VM stuurt

#### RF-DETR Model

- **Model:** RFDETRBase (Real-time Detection Transformer)
- **Training:** Pre-trained op COCO dataset
- **Classes gebruikt:**
  | COCO ID | Label |
  |---------|-------|
  | 2 | bicycle |
  | 3 | car |
  | 4 | motorcycle |
  | 6 | bus |
  | 8 | truck |

#### Configuratie

```python
# camera_to_server/detect_cars_to_server.py
CAMERA_DEVICE = "/dev/video1"
SAMPLE_EVERY_SECONDS = 10.0
VM_API_URL = "http://<VM_IP>:8001/api/v1/observations"
CAMERA_ID = "rock5-camera-1"
```

#### Opstarten Rock5

```bash
# Op de Rock5
cd ~/AI-Traffic-detector/camera_to_server

# Docker build en run
docker build -t traffic-camera .
docker run -d \
  --device=/dev/video1 \
  --name traffic-camera \
  traffic-camera

# Of zonder Docker
pip install -r requirements.txt
python detect_cars_to_server.py
```

---

### VM Backend Server

**Locatie:** `VM-side/`

De server ontvangt data van de Rock5, slaat het op in PostgreSQL, en serveert de API + dashboard.

#### Tech Stack

- **Framework:** FastAPI
- **Database:** PostgreSQL 16
- **ML:** PyTorch met CUDA support
- **Container:** Docker met NVIDIA GPU runtime

#### Database Schema

```sql
CREATE TABLE traffic_samples (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP WITH TIME ZONE NOT NULL,
    camera_id VARCHAR,
    total_vehicles INTEGER DEFAULT 0,
    car INTEGER DEFAULT 0,
    truck INTEGER DEFAULT 0,
    bus INTEGER DEFAULT 0,
    motorcycle INTEGER DEFAULT 0,
    bicycle INTEGER DEFAULT 0,
    snapshot_path VARCHAR
);
```

#### Opstarten VM Server

```bash
# Op de VM
cd ~/AI-Traffic-detector

# Start met Docker Compose
docker compose up -d --build

# Logs bekijken
docker compose logs -f api

# Interface openen
# http://<VM_IP>:8001/
```

#### Environment Variables

| Variable | Default | Beschrijving |
|----------|---------|--------------|
| `DATABASE_URL` | `postgresql+psycopg2://traffic_user:supersecretpassword@db/traffic_db` | PostgreSQL connectie string |

---

### Transformer Prediction Model

**Locatie:** `VM-side/transformer_model/`

Een encoder-only Transformer model dat verkeerspatronen leert en voorspellingen maakt.

#### Model Architectuur

```
Input (24h) ──▶ Linear Embedding ──▶ Positional Encoding
                                            │
                                            ▼
                                   Transformer Encoder
                                   (2 layers, 4 heads)
                                            │
                                            ▼
                                   Flatten + MLP
                                            │
                                            ▼
                                   Output (24h predictions)
```

#### Hyperparameters

```python
# transformer_model/config.py
MODEL_CONFIG = {
    "input_dim": 4,           # [car_normalized, hour, day_of_week, is_weekend]
    "output_dim": 1,          # predicted car count
    "d_model": 64,            # transformer dimension
    "nhead": 4,               # attention heads
    "num_encoder_layers": 2,  # encoder layers
    "dim_feedforward": 128,   # feedforward dimension
    "dropout": 0.1,
    "seq_len": 24,            # input: 24 uur
    "pred_len": 24,           # output: 24 uur
}

TRAIN_CONFIG = {
    "batch_size": 4,
    "epochs": 150,
    "learning_rate": 0.0005,
    "train_split": 0.8,
    "early_stopping_patience": 15,
}
```

#### Features

Het model gebruikt 4 input features per tijdstip:

| Feature | Normalisatie | Beschrijving |
|---------|--------------|--------------|
| `car` | MinMaxScaler (0-1) | Gemiddeld aantal auto's per snapshot |
| `hour` | /23 | Uur van de dag (0-23) |
| `day_of_week` | /6 | Dag van de week (0=ma, 6=zo) |
| `is_weekend` | 0 of 1 | Weekend indicator |

#### Training

```bash
# In de Docker container
docker exec -it traffic_api python -c "
from transformer_model.train import train_model
train_model(epochs=150, verbose=True)
"

# Of via Python
cd VM-side
python -m transformer_model.train
```

#### Voorspelling Flow

1. **Data ophalen:** Laatste 24 uur uit database
2. **Aggregatie:** Gemiddelde per uur (mean, niet sum)
3. **Features:** Temporal features toevoegen
4. **Normalisatie:** MinMaxScaler met 99e percentiel clipping
5. **Inference:** Transformer forward pass
6. **Denormalisatie:** Inverse transform
7. **Post-processing:** Dag-specifieke correctie

---

## 🚀 Installatie & Opstarten

### Vereisten

- Docker & Docker Compose
- NVIDIA GPU met CUDA 12.1+ (voor VM)
- Rock5 board met USB camera

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-repo/AI-Traffic-detector.git
cd AI-Traffic-detector

# Start VM server
docker compose up -d --build

# Wacht tot database klaar is
sleep 10

# Check status
docker compose ps

# Open dashboard
xdg-open http://localhost:8001/
```

### Volledige Setup

#### 1. VM Server

```bash
# Op de VM
cd AI-Traffic-detector
docker compose up -d --build
```

#### 2. Rock5 Camera

```bash
# Op de Rock5
cd AI-Traffic-detector/camera_to_server

# Edit VM IP adres
nano detect_cars_to_server.py
# Wijzig: VM_API_URL = "http://<VM_IP>:8001/api/v1/observations"

# Start
docker build -t traffic-camera .
docker run -d --device=/dev/video1 --restart=unless-stopped traffic-camera
```

#### 3. Model Trainen

```bash
# Verzamel eerst data (minimaal 48+ uur)
# Train het model
docker exec traffic_api python -c "
from transformer_model.train import train_model
train_model()
"
```

---

## 📡 API Endpoints

### Base URL: `http://<VM_IP>:8001`

### Status & Monitoring

#### `GET /api/v1/status`
Haal systeem status en laatste observaties op.

**Response:**
```json
{
  "server_time": "2025-12-16T14:30:00+00:00",
  "rock5_status": "online",
  "vm_status": "online",
  "db_status": "ok",
  "sample_age_seconds": 5.2,
  "last_observations": [
    {
      "id": 4521,
      "ts": "2025-12-16T14:29:55+00:00",
      "camera_id": "rock5-camera-1",
      "total_vehicles": 12,
      "car": 10,
      "truck": 1,
      "bus": 0,
      "motorcycle": 1,
      "bicycle": 0,
      "snapshot_url": "/static/snapshots/xxx.jpg"
    }
  ]
}
```

### Observations

#### `POST /api/v1/observations`
Ontvang nieuwe observatie van camera (multipart/form-data).

**Form Fields:**
- `payload` (string, JSON): Observatie data
- `snapshot` (file, optional): JPEG/PNG afbeelding

**Payload JSON:**
```json
{
  "ts": "2025-12-16T14:30:00Z",
  "camera_id": "rock5-camera-1",
  "total_vehicles": 5,
  "breakdown": {
    "car": 4,
    "truck": 1,
    "bus": 0,
    "motorcycle": 0,
    "bicycle": 0
  }
}
```

#### `POST /api/v1/observation`
Simplified endpoint voor debug/testing.

**Body (JSON):**
```json
{
  "ts": "2025-12-16T14:30:00Z",
  "camera_id": "test",
  "total_vehicles": 5,
  "car": 4,
  "truck": 1
}
```

### Predictions

#### `GET /api/v1/predictions`
Haal voorspellingen op voor vandaag of specifieke datum.

**Query Parameters:**
- `date` (optional): Datum in `YYYY-MM-DD` formaat

**Response:**
```json
{
  "predictions": [
    {
      "hour": "2025-12-16 00:00",
      "hour_of_day": 0,
      "predicted_cars": 1,
      "is_rush_hour": false
    },
    {
      "hour": "2025-12-16 07:00",
      "hour_of_day": 7,
      "predicted_cars": 24,
      "is_rush_hour": true
    }
  ],
  "summary": {
    "total_predicted": 245,
    "average_per_hour": 10.2,
    "peak_hour": "2025-12-16 08:00",
    "peak_cars": 29,
    "rush_hour_average": 24.5
  },
  "model_ready": true,
  "value_unit": "avg vehicles per snapshot (hourly block)",
  "value_note": "Not an hourly total; reflects average vehicles visible per snapshot"
}
```

#### `GET /api/v1/predictions/current`
Voorspelling voor het huidige uur.

#### `GET /api/v1/predictions/week`
Voorspellingen voor de komende 7 dagen.

### Model Metrics

#### `GET /api/v1/model/metrics`
Validatie metrics van het model.

**Response:**
```json
{
  "mae": 2.15,
  "rmse": 3.42,
  "smape": 18.5,
  "per_hour_mae": [0.5, 0.4, 0.3, ...],
  "samples": 1200,
  "sequences": 50,
  "model_ready": true
}
```

---

## 📊 Dashboard Interface

**URL:** `http://<VM_IP>:8001/`

### Monitoring Tab

- **Last 10 Observations:** Recente tellingen met snapshot preview
- **System Status:**
  - Rock5 camera pipeline status (online/lagging/offline)
  - FastAPI service status
  - Database status
  - Sample age in seconden

### Predictions Tab

- **Traffic Forecast Chart:** Bar chart met 24-uurs voorspellingen
- **Day Selector:** Kies datum (vandaag + 6 dagen vooruit)
- **Current Hour:** Live voorspelling voor dit uur
- **Summary Stats:** Piek uur, daggemiddelde, spitsuur gemiddelde
- **Model Evaluation:** MAE, RMSE, sMAPE metrics met per-uur breakdown

---

## 📈 Model Evaluatie

### Metrics Uitleg

| Metric | Betekenis | Goed als |
|--------|-----------|----------|
| **MAE** | Mean Absolute Error - gemiddelde absolute afwijking | < 5 |
| **RMSE** | Root Mean Squared Error - straft uitschieters zwaarder | < 8 |
| **sMAPE** | Symmetric MAPE - schaal-invariant percentage | < 25% |

### Huidige Prestaties

```
Overall MAE: ~2.1 auto's per uur
Spitsuur nauwkeurigheid: ±1-3 auto's
Weekend vs Weekday: Correct onderscheid ✓
```

### Per Dag van de Week

| Dag | MAE | Status |
|-----|-----|--------|
| Maandag | ~1.1 | ✅ Uitstekend |
| Dinsdag | ~1.1 | ✅ Uitstekend |
| Woensdag | ~1.0 | ✅ Uitstekend |
| Donderdag | ~1.2 | ✅ Uitstekend |
| Vrijdag | ~6.5 | ⚠️ Matig |
| Zaterdag | ~0.4 | ✅ Uitstekend |
| Zondag | ~0.4 | ✅ Uitstekend |

---

## 📁 Project Structuur

```
AI-Traffic-detector/
├── README.md                    # Deze file
├── docker-compose.yml           # VM orchestration
│
├── camera_to_server/            # Rock5 edge code
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── detect_cars_to_server.py # Hoofdscript met RF-DETR
│   └── readme.md
│
├── VM-side/                     # Server code
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # FastAPI applicatie
│   ├── static/
│   │   ├── dashboard.html       # Dashboard UI
│   │   ├── dashboard.css        # Styles
│   │   └── dashboard.js         # JavaScript logic
│   └── transformer_model/
│       ├── __init__.py
│       ├── config.py            # Hyperparameters
│       ├── model.py             # Transformer architectuur
│       ├── data_preprocessing.py # Data pipeline
│       ├── train.py             # Training script
│       └── predict.py           # Inference
│
├── Evaluation model/
│   └── evaluate_model.py        # Standalone evaluatie
│
└── debug/
    └── generate_realistic_data.py # Test data generator
```

---

## 🔧 Troubleshooting

### Camera niet gevonden
```bash
# Check devices
ls -la /dev/video*

# Test camera
ffmpeg -f v4l2 -i /dev/video1 -frames:v 1 test.jpg
```

### Model niet getraind
```bash
# Check of er genoeg data is (minimaal 48 uur)
docker exec traffic_api python -c "
from transformer_model.data_preprocessing import load_data_from_db
df = load_data_from_db()
print(f'Samples: {len(df)}')
"
```

### GPU niet beschikbaar
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
docker exec traffic_api python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
"
```

---

## � CI/CD Pipeline

Dit project gebruikt GitHub Actions voor automatische testing en deployment.

### Workflows

#### 1. CI/CD Pipeline (`ci.yml`)

Wordt automatisch uitgevoerd bij elke push en pull request.

| Stage | Beschrijving |
|-------|--------------|
| 🔍 **Lint** | Code quality checks met flake8 en black |
| 🧪 **Test API** | FastAPI imports, model architectuur tests |
| 📷 **Test Camera** | Syntax check van camera script |
| 🐳 **Build Docker** | Build beide Docker images |
| 🔗 **Integration** | Start docker-compose, test endpoints |
| 🚀 **Deploy** | Deploy naar server (na merge naar main) |

#### 2. Model Training (`train.yml`)

Handmatig te starten of wekelijks (zondag 2:00 UTC).

```bash
# Handmatig starten via GitHub UI:
# Actions → Model Training & Evaluation → Run workflow
```

### Deployment Configureren

Om automatische deployment in te schakelen, voeg deze secrets toe in GitHub:

1. Ga naar **Settings** → **Secrets and variables** → **Actions**
2. Voeg toe:
   - `SSH_PRIVATE_KEY`: SSH private key voor server toegang
   - `SERVER_HOST`: Server IP (bijv. `100.89.11.82`)
   - `SERVER_USER`: SSH gebruiker (bijv. `root`)

### Lokaal Testen

```bash
# Install act voor lokale GitHub Actions testing
# https://github.com/nektos/act

# Run CI lokaal
act push
```

---

## 📝 Licentie

MIT License

---
