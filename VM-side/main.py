import os
import pathlib
from datetime import datetime, timezone
from typing import Optional, List, Literal

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
)
from sqlalchemy.orm import sessionmaker, Session, declarative_base


# ---------------------------------------------------------------------------
# Config & DB setup
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    # Default voor in Docker (host: traffic_db) – kan via env worden override'd
    "postgresql+psycopg2://traffic_user:supersecretpassword@traffic_db/traffic_db",
)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TrafficSample(Base):
    __tablename__ = "traffic_samples"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), index=True, nullable=False)

    total_vehicles = Column(Integer, nullable=False, default=0)
    car = Column(Integer, nullable=False, default=0)
    truck = Column(Integer, nullable=False, default=0)
    bus = Column(Integer, nullable=False, default=0)
    motorcycle = Column(Integer, nullable=False, default=0)
    bicycle = Column(Integer, nullable=False, default=0)

    # Je houdt camera_id – ok, 1 camera maar future-proof
    camera_id = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ObservationIn(BaseModel):
    timestamp: Optional[datetime] = None
    total_vehicles: int
    car: int = 0
    truck: int = 0
    bus: int = 0
    motorcycle: int = 0
    bicycle: int = 0
    camera_id: Optional[str] = "rock5-1"


class ObservationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: datetime
    total_vehicles: int
    car: int
    truck: int
    bus: int
    motorcycle: int
    bicycle: int
    camera_id: Optional[str]


class DashboardStatus(BaseModel):
    vm_status: str
    db_status: str
    rock5_status: Literal["online", "lagging", "offline"]
    server_time: datetime
    sample_age_seconds: Optional[float]
    last_observations: List[ObservationOut]


# ---------------------------------------------------------------------------
# FastAPI app + static mounting
# ---------------------------------------------------------------------------

app = FastAPI(title="Traffic Counter API (1 camera)")

BASE_DIR = pathlib.Path(__file__).parent

# static/ folder (voor dashboard.html)
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    """Serve de futuristic dashboard UI."""
    index_file = BASE_DIR / "static" / "dashboard.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return index_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# API: Rock5 → POST observaties
# ---------------------------------------------------------------------------

@app.post("/api/v1/observations", response_model=ObservationOut)
def create_observation(payload: ObservationIn, db: Session = Depends(get_db)):
    # timestamp van de Rock5 of fallback naar server-tijd (UTC)
    if payload.timestamp is None:
        ts = datetime.now(timezone.utc)
    else:
        ts = payload.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

    row = TrafficSample(
        ts=ts,
        total_vehicles=payload.total_vehicles,
        car=payload.car,
        truck=payload.truck,
        bus=payload.bus,
        motorcycle=payload.motorcycle,
        bicycle=payload.bicycle,
        camera_id=payload.camera_id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ---------------------------------------------------------------------------
# API: Status + laatste 10 metingen
# ---------------------------------------------------------------------------

ROCK5_OFFLINE_THRESHOLD_SEC = 120  # >2 min geen nieuwe sample = offline


@app.get("/api/v1/status", response_model=DashboardStatus)
def get_status(db: Session = Depends(get_db)):
    rows: List[TrafficSample] = (
        db.query(TrafficSample)
        .order_by(TrafficSample.ts.desc())
        .limit(10)
        .all()
    )

    now = datetime.now(timezone.utc)

    if rows:
        last_ts = rows[0].ts
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        age_sec: Optional[float] = (now - last_ts).total_seconds()
    else:
        last_ts = None
        age_sec = None

    # Rock5 status op basis van leeftijd laatste sample
    if last_ts is None:
        rock5_status: Literal["online", "lagging", "offline"] = "offline"
    else:
        if age_sec < 30:
            rock5_status = "online"
        elif age_sec < ROCK5_OFFLINE_THRESHOLD_SEC:
            rock5_status = "lagging"
        else:
            rock5_status = "offline"

    vm_status = "online"
    db_status = "online"

    return DashboardStatus(
        vm_status=vm_status,
        db_status=db_status,
        rock5_status=rock5_status,
        server_time=now,
        sample_age_seconds=age_sec,
        last_observations=rows,
    )
