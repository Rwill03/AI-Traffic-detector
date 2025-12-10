import os
import json
import uuid
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Form, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import OperationalError

# Import predictor (lazy loading)
from transformer_model.predict import get_predictor, predict_next_24_hours, predict_for_date

# === Paths & DB setup ===

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SNAPSHOT_DIR = STATIC_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://traffic_user:supersecretpassword@db/traffic_db",
)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TrafficSample(Base):
    __tablename__ = "traffic_samples"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), index=True, nullable=False)
    camera_id = Column(String, nullable=True)

    total_vehicles = Column(Integer, nullable=False, default=0)
    car = Column(Integer, nullable=False, default=0)
    truck = Column(Integer, nullable=False, default=0)
    bus = Column(Integer, nullable=False, default=0)
    motorcycle = Column(Integer, nullable=False, default=0)
    bicycle = Column(Integer, nullable=False, default=0)

    # nieuw: pad naar snapshot (relative URL, bv. "/static/snapshots/xxx.jpg")
    snapshot_path = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# === FastAPI app ===

app = FastAPI(title="Traffic Counter API (1 camera)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static + dashboard
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_root():
    dashboard = STATIC_DIR / "dashboard.html"
    if not dashboard.exists():
        raise HTTPException(500, detail="dashboard.html not found")
    return FileResponse(str(dashboard))


# === Helpers ===


def parse_ts(value: str | None) -> datetime:
    """Parse timestamp, gebruik server tijd als de timestamp in de toekomst ligt."""
    now = datetime.now(timezone.utc)
    
    if not value:
        return now
    
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Als timestamp meer dan 1 minuut in de toekomst ligt, gebruik server tijd
        if dt > now + timedelta(minutes=1):
            print(f"Warning: Camera timestamp {dt} is in the future, using server time instead")
            return now
        
        return dt
    except ValueError:
        return datetime.now(timezone.utc)


def sample_to_dict(s: TrafficSample) -> dict:
    return {
        "id": s.id,
        "ts": s.ts.isoformat(),
        "camera_id": s.camera_id,
        "total_vehicles": s.total_vehicles,
        "car": s.car,
        "truck": s.truck,
        "bus": s.bus,
        "motorcycle": s.motorcycle,
        "bicycle": s.bicycle,
        "snapshot_url": s.snapshot_path,  # front-end verwacht dit veld
    }


def compute_rock5_status(latest: TrafficSample | None) -> tuple[str, float | None]:
    if latest is None:
        return "offline", None
    now = datetime.now(timezone.utc)
    age = (now - latest.ts).total_seconds()

    # zelfde logica als vroeger: < 30s = online, < 120s = lagging, anders offline
    if age < 30:
        return "online", age
    elif age < 120:
        return "lagging", age
    else:
        return "offline", age


# === API endpoints ===


@app.get("/api/v1/status")
def get_status(db: Session = Depends(get_db)):
    server_time = datetime.now(timezone.utc)

    try:
        # laatste 10 samples voor de tabel
        samples = (
            db.query(TrafficSample)
            .order_by(TrafficSample.ts.desc())
            .limit(10)
            .all()
        )
        last = samples[0] if samples else None

        rock5_status, age = compute_rock5_status(last)
        vm_status = "online"
        db_status = "ok"
    except OperationalError:
        samples = []
        rock5_status = "offline"
        age = None
        vm_status = "degraded"
        db_status = "error"

    return {
        "server_time": server_time.isoformat(),
        "rock5_status": rock5_status,
        "vm_status": vm_status,
        "db_status": db_status,
        "sample_age_seconds": age,
        "last_observations": [sample_to_dict(s) for s in samples],
    }


@app.post("/api/v1/observations")
async def create_observation(
    # Let op: multipart/form-data
    payload: str = Form(...),
    snapshot: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """
    Verwacht:
      - payload (Form field, JSON string):
          {
            "ts": "...",            # optioneel, ISO
            "camera_id": "rock5-1", # optioneel
            "total_vehicles": 4,
            "breakdown": {
              "car": 3,
              "truck": 1,
              ...
            }
          }
      - snapshot (optional file): JPEG/PNG met boxes
    """

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'payload' field")

    # prefer nested breakdown; fall back to legacy top-level keys if present
    ts_raw = data.get("ts") or data.get("timestamp")
    ts = parse_ts(ts_raw)
    breakdown = data.get("breakdown") or {}
    for key in ("car", "truck", "bus", "motorcycle", "bicycle"):
        if key not in breakdown and key in data:
            breakdown[key] = data[key]

    # snapshot opslaan (indien aanwezig)
    snapshot_rel_url: str | None = None
    if snapshot is not None:
        # bepaal extensie
        orig_name = snapshot.filename or "snapshot.jpg"
        ext = Path(orig_name).suffix.lower()
        if not ext:
            ext = ".jpg"

        filename = f"{int(ts.timestamp() * 1000)}_{uuid.uuid4().hex}{ext}"
        out_path = SNAPSHOT_DIR / filename

        content = await snapshot.read()
        with out_path.open("wb") as f:
            f.write(content)

        # relative URL
        snapshot_rel_url = f"/static/snapshots/{filename}"

    sample = TrafficSample(
        ts=ts,
        camera_id=data.get("camera_id"),
        total_vehicles=int(data.get("total_vehicles") or 0),
        car=int(breakdown.get("car") or 0),
        truck=int(breakdown.get("truck") or 0),
        bus=int(breakdown.get("bus") or 0),
        motorcycle=int(breakdown.get("motorcycle") or 0),
        bicycle=int(breakdown.get("bicycle") or 0),
        snapshot_path=snapshot_rel_url,
    )

    db.add(sample)
    db.commit()
    db.refresh(sample)

    return {
        "ok": True,
        "id": sample.id,
        "snapshot_url": snapshot_rel_url,
    }


# === Simple observation endpoint (for debug data generator) ===

@app.post("/api/v1/observation")
async def create_simple_observation(
    data: dict,
    db: Session = Depends(get_db),
):
    """
    Simplified observation endpoint voor het debug data generator script.
    Verwacht een JSON body met ts, camera_id, total_vehicles, car, truck, bus, motorcycle, bicycle.
    """
    ts = parse_ts(data.get("ts"))
    
    sample = TrafficSample(
        ts=ts,
        camera_id=data.get("camera_id"),
        total_vehicles=int(data.get("total_vehicles") or 0),
        car=int(data.get("car") or 0),
        truck=int(data.get("truck") or 0),
        bus=int(data.get("bus") or 0),
        motorcycle=int(data.get("motorcycle") or 0),
        bicycle=int(data.get("bicycle") or 0),
        snapshot_path=None,
    )

    db.add(sample)
    db.commit()
    db.refresh(sample)

    return {"ok": True, "id": sample.id}


# === Prediction API endpoints ===

@app.get("/api/v1/predictions")
def get_predictions(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
):
    """
    Haal voorspellingen op voor de volgende 24 uur of voor een specifieke datum.
    
    Args:
        date: Optionele datum in YYYY-MM-DD formaat
        
    Returns:
        Dict met predictions en summary
    """
    predictor = get_predictor()
    
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            predictions = predict_for_date(target_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        predictions = predict_next_24_hours()
    
    summary = predictor.get_daily_summary(predictions)
    
    return {
        "predictions": predictions,
        "summary": summary,
        "model_ready": predictor.is_ready(),
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/v1/predictions/current")
def get_current_prediction():
    """
    Haal de voorspelling voor het huidige uur op.
    """
    predictor = get_predictor()
    current = predictor.get_current_hour_prediction()
    
    if current is None:
        raise HTTPException(status_code=500, detail="Could not generate prediction")
    
    return {
        "current_hour": current,
        "model_ready": predictor.is_ready(),
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/v1/predictions/week")
def get_week_predictions():
    """
    Haal voorspellingen op voor de komende 7 dagen.
    """
    predictor = get_predictor()
    today = datetime.now()
    
    week_predictions = []
    for i in range(7):
        target_date = today + timedelta(days=i)
        predictions = predict_for_date(target_date)
        summary = predictor.get_daily_summary(predictions)
        
        week_predictions.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "day_name": target_date.strftime("%A"),
            "predictions": predictions,
            "summary": summary
        })
    
    return {
        "week_predictions": week_predictions,
        "model_ready": predictor.is_ready(),
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
