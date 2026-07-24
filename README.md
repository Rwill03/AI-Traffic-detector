# 🚗 AI Traffic Detector

A real-time traffic monitoring system that combines AI-driven vehicle detection with a Transformer model for traffic predictions.

![Architecture](https://img.shields.io/badge/Architecture-Edge%20%2B%20Cloud-blue)
![Model](https://img.shields.io/badge/Detection-RF--DETR-green)
![Prediction](https://img.shields.io/badge/Prediction-Transformer-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![CI/CD](https://github.com/Rwill03/AI-Traffic-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/Rwill03/AI-Traffic-detector/actions/workflows/ci.yml)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
  - [Rock5 Edge Device](#rock5-edge-device-camera-detection)
  - [VM Backend Server](#vm-backend-server)
  - [Transformer Prediction Model](#transformer-prediction-model)
- [Installation & Startup](#installation--startup)
- [API Endpoints](#api-endpoints)
- [Dashboard Interface](#dashboard-interface)
- [Model Evaluation](#model-evaluation)
- [CI/CD Pipeline](#cicd-pipeline)

---

## 🎯 Overview

This system detects and counts vehicles in real-time via a camera connected to a Rock5 edge device. The data is forwarded to a VM server with GPU, where a Transformer model learns traffic patterns and makes predictions for the next 24 hours.

### Features

- ✅ **Real-time vehicle detection** with RF-DETR (COCO dataset)
- ✅ **Classification** by vehicle type: car, truck, bus, motorcycle, bicycle
- ✅ **24-hour predictions** with AI Transformer model
- ✅ **Weekend/weekday awareness** in predictions
- ✅ **Rush hour detection** (7-9 AM and 4-6 PM)
- ✅ **Live dashboard** with monitoring and predictions
- ✅ **GPU acceleration** for fast inference

---

## 🏗️ Architecture

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
│                              ▼  HTTP POST (every 10s)                      │
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
│  │  Dashboard  │  ◀── Real-time updates every 5s                        ││
│  │  (Web UI)   │                                                        ││
│  └─────────────┘                                                        ││
└─────────────────────────────────────────────────────────────────────────┘│
```

---

## 📦 Components

### Rock5 Edge Device (Camera Detection)

**Location:** `camera_to_server/`

The Rock5 board runs a headless Python script that:
1. Reads frames from the USB camera (`/dev/video1`)
2. Performs inference every 10 seconds with RF-DETR
3. Classifies and counts vehicles
4. Sends snapshot + counts to the VM

#### RF-DETR Model

- **Model:** RFDETRBase (Real-time Detection Transformer)
- **Training:** Pre-trained on COCO dataset
- **Classes used:**
  | COCO ID | Label |
  |---------|-------|
  | 2 | bicycle |
  | 3 | car |
  | 4 | motorcycle |
  | 6 | bus |
  | 8 | truck |

#### Configuration

```python
# camera_to_server/detect_cars_to_server.py
CAMERA_DEVICE = "/dev/video1"
SAMPLE_EVERY_SECONDS = 10.0
VM_API_URL = "http://<VM_IP>:8001/api/v1/observations"
CAMERA_ID = "rock5-camera-1"
```

#### Starting Rock5

```bash
# On the Rock5
cd ~/AI-Traffic-detector/camera_to_server

# Docker build and run
docker build -t traffic-camera .
docker run -d \
  --device=/dev/video1 \
  --name traffic-camera \
  traffic-camera

# Or without Docker
pip install -r requirements.txt
python detect_cars_to_server.py
```

---

### VM Backend Server

**Location:** `VM-side/`

The server receives data from the Rock5, stores it in PostgreSQL, and serves the API + dashboard.

#### Tech Stack

- **Framework:** FastAPI
- **Database:** PostgreSQL 16
- **ML:** PyTorch with CUDA support
- **Container:** Docker with NVIDIA GPU runtime

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

#### Starting VM Server

```bash
# On the VM
cd ~/AI-Traffic-detector

# Start with Docker Compose
docker compose up -d --build

# View logs
docker compose logs -f api

# Open interface
# http://<VM_IP>:8001/
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+psycopg2://traffic_user:supersecretpassword@db/traffic_db` | PostgreSQL connection string |

---

### Transformer Prediction Model

**Location:** `VM-side/transformer_model/`

An encoder-only Transformer model that learns traffic patterns and makes predictions.

#### Model Architecture

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

The model uses 4 input features per time step:

| Feature | Normalization | Description |
|---------|---|---|
| `car` | MinMaxScaler (0-1) | Average number of cars per snapshot |
| `hour` | /23 | Hour of day (0-23) |
| `day_of_week` | /6 | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | 0 or 1 | Weekend indicator |

#### Training

```bash
# In the Docker container
docker exec -it traffic_api python -c "
from transformer_model.train import train_model
train_model(epochs=150, verbose=True)
"

# Or via Python
cd VM-side
python -m transformer_model.train
```

#### Prediction Flow

1. **Fetch data:** Last 24 hours from database
2. **Aggregation:** Average per hour (mean, not sum)
3. **Features:** Add temporal features
4. **Normalization:** MinMaxScaler with 99th percentile clipping
5. **Inference:** Transformer forward pass
6. **Denormalization:** Inverse transform
7. **Post-processing:** Day-specific correction

---

## 🚀 Installation & Startup

### Requirements

- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.1+ (for VM)
- Rock5 board with USB camera

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-repo/AI-Traffic-detector.git
cd AI-Traffic-detector

# Start VM server
docker compose up -d --build

# Wait until database is ready
sleep 10

# Check status
docker compose ps

# Open dashboard
xdg-open http://localhost:8001/
```

### Full Setup

#### 1. VM Server

```bash
# On the VM
cd AI-Traffic-detector
docker compose up -d --build
```

#### 2. Rock5 Camera

```bash
# On the Rock5
cd AI-Traffic-detector/camera_to_server

# Edit VM IP address
nano detect_cars_to_server.py
# Change: VM_API_URL = "http://<VM_IP>:8001/api/v1/observations"

# Start
docker build -t traffic-camera .
docker run -d --device=/dev/video1 --restart=unless-stopped traffic-camera
```

#### 3. Model Training

```bash
# Collect data first (minimum 48+ hours)
# Train the model
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
Fetch system status and latest observations.

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
Receive new observation from camera (multipart/form-data).

**Form Fields:**
- `payload` (string, JSON): Observation data
- `snapshot` (file, optional): JPEG/PNG image

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
Simplified endpoint for debug/testing.

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
Fetch predictions for today or specific date.

**Query Parameters:**
- `date` (optional): Date in `YYYY-MM-DD` format

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
Prediction for the current hour.

#### `GET /api/v1/predictions/week`
Predictions for the next 7 days.

### Model Metrics

#### `GET /api/v1/model/metrics`
Model validation metrics.

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

- **Last 10 Observations:** Recent counts with snapshot preview
- **System Status:**
  - Rock5 camera pipeline status (online/lagging/offline)
  - FastAPI service status
  - Database status
  - Sample age in seconds

### Predictions Tab

- **Traffic Forecast Chart:** Bar chart with 24-hour predictions
- **Day Selector:** Select date (today + 6 days ahead)
- **Current Hour:** Live prediction for this hour
- **Summary Stats:** Peak hour, daily average, rush hour average
- **Model Evaluation:** MAE, RMSE, sMAPE metrics with per-hour breakdown

---

## 📈 Model Evaluation

### Metrics Explanation

| Metric | Meaning | Good if |
|--------|---------|--------|
| **MAE** | Mean Absolute Error - average absolute deviation | < 5 |
| **RMSE** | Root Mean Squared Error - penalizes outliers harder | < 8 |
| **sMAPE** | Symmetric MAPE - scale-invariant percentage | < 25% |

### Current Performance

```
Overall MAE: ~2.1 cars per hour
Rush hour accuracy: ±1-3 cars
Weekend vs Weekday: Correct distinction ✓
```

### Per Day of Week

| Day | MAE | Status |
|-----|-----|--------|
| Monday | ~1.1 | ✅ Excellent |
| Tuesday | ~1.1 | ✅ Excellent |
| Wednesday | ~1.0 | ✅ Excellent |
| Thursday | ~1.2 | ✅ Excellent |
| Friday | ~6.5 | ⚠️ Fair |
| Saturday | ~0.4 | ✅ Excellent |
| Sunday | ~0.4 | ✅ Excellent |

---

## 📁 Project Structure

```
AI-Traffic-detector/
├── README.md                    # This file
├── docker-compose.yml           # VM orchestration
│
├── camera_to_server/            # Rock5 edge code
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── detect_cars_to_server.py # Main script with RF-DETR
│   └── readme.md
│
├── VM-side/                     # Server code
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # FastAPI application
│   ├── static/
│   │   ├── dashboard.html       # Dashboard UI
│   │   ├── dashboard.css        # Styles
│   │   └── dashboard.js         # JavaScript logic
│   └── transformer_model/
│       ├── __init__.py
│       ├── config.py            # Hyperparameters
│       ├── model.py             # Transformer architecture
│       ├── data_preprocessing.py # Data pipeline
│       ├── train.py             # Training script
│       └── predict.py           # Inference
│
├── Evaluation model/
│   └── evaluate_model.py        # Standalone evaluation
│
└── debug/
    └── generate_realistic_data.py # Test data generator
```

---

## 🔧 Troubleshooting

### Camera not found
```bash
# Check devices
ls -la /dev/video*

# Test camera
ffmpeg -f v4l2 -i /dev/video1 -frames:v 1 test.jpg
```

### Model not trained
```bash
# Check if there's enough data (minimum 48 hours)
docker exec traffic_api python -c "
from transformer_model.data_preprocessing import load_data_from_db
df = load_data_from_db()
print(f'Samples: {len(df)}')
"
```

### GPU not available
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

This project uses GitHub Actions for automated testing and deployment.

### Workflows

#### 1. CI/CD Pipeline (`ci.yml`)

Runs automatically on every push and pull request.

| Stage | Description |
|-------|-------------|
| 🔍 **Lint** | Code quality checks with flake8 and black |
| 🧪 **Test API** | FastAPI imports, model architecture tests |
| 📷 **Test Camera** | Syntax check of camera script |
| 🐳 **Build Docker** | Build both Docker images |
| 🔗 **Integration** | Start docker-compose, test endpoints |
| 🚀 **Deploy** | Deploy to server (after merge to main) |

#### 2. Model Training (`train.yml`)

Manually triggered or weekly (Sunday 2:00 UTC).

```bash
# Manually start via GitHub UI:
# Actions → Model Training & Evaluation → Run workflow
```

### Configure Deployment

To enable automatic deployment, add these secrets in GitHub:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add:
   - `SSH_PRIVATE_KEY`: SSH private key for server access
   - `SERVER_HOST`: Server IP (e.g., `100.89.11.82`)
   - `SERVER_USER`: SSH user (e.g., `root`)

### Local Testing

```bash
# Install act for local GitHub Actions testing
# https://github.com/nektos/act

# Run CI locally
act push
```

---

## 📝 License

MIT License

---
