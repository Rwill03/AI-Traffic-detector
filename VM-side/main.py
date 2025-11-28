from datetime import datetime
from typing import List

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ===== DB setup =====
DATABASE_URL = "postgresql+psycopg2://traffic_user:supersecretpassword@localhost/traffic_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TrafficSample(Base):
    __tablename__ = "traffic_samples"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, index=True, nullable=False)
    total_vehicles = Column(Integer, nullable=False)
    car = Column(Integer, nullable=False)
    truck = Column(Integer, nullable=False)
    bus = Column(Integer, nullable=False)
    motorcycle = Column(Integer, nullable=False)
    bicycle = Column(Integer, nullable=False)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===== Pydantic models =====
class TrafficSampleIn(BaseModel):
    timestamp: datetime = Field(..., example="2025-11-28T17:25:00")
    total_vehicles: int = Field(..., ge=0)
    car: int = Field(0, ge=0)
    truck: int = Field(0, ge=0)
    bus: int = Field(0, ge=0)
    motorcycle: int = Field(0, ge=0)
    bicycle: int = Field(0, ge=0)


class TrafficSampleOut(BaseModel):
    id: int
    timestamp: datetime
    total_vehicles: int
    car: int
    truck: int
    bus: int
    motorcycle: int
    bicycle: int

    class Config:
        orm_mode = True


app = FastAPI(title="Traffic Counter API (1 camera)")


@app.post("/api/v1/observations", response_model=TrafficSampleOut)
def create_observation(
    sample: TrafficSampleIn,
    db: Session = Depends(get_db),
):
    db_obj = TrafficSample(
        ts=sample.timestamp,
        total_vehicles=sample.total_vehicles,
        car=sample.car,
        truck=sample.truck,
        bus=sample.bus,
        motorcycle=sample.motorcycle,
        bicycle=sample.bicycle,
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return TrafficSampleOut(
        id=db_obj.id,
        timestamp=db_obj.ts,
        total_vehicles=db_obj.total_vehicles,
        car=db_obj.car,
        truck=db_obj.truck,
        bus=db_obj.bus,
        motorcycle=db_obj.motorcycle,
        bicycle=db_obj.bicycle,
    )


@app.get("/api/v1/observations", response_model=List[TrafficSampleOut])
def list_observations(
    limit: int = 100,
    db: Session = Depends(get_db),
):
    rows = (
        db.query(TrafficSample)
        .order_by(TrafficSample.ts.desc())
        .limit(limit)
        .all()
    )
    return [
        TrafficSampleOut(
            id=r.id,
            timestamp=r.ts,
            total_vehicles=r.total_vehicles,
            car=r.car,
            truck=r.truck,
            bus=r.bus,
            motorcycle=r.motorcycle,
            bicycle=r.bicycle,
        )
        for r in rows
    ]
