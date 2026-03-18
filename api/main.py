"""FastAPI REST wrapper around opl-py for the prediction UI."""

from __future__ import annotations

import datetime
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import opl
from opl.analytics import (
    BaseTrajectoryModel,
    get_all_approaches,
)

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
_client: opl.OPL | None = None
_models: dict[str, BaseTrajectoryModel] = {}


def _get_client() -> opl.OPL:
    assert _client is not None, "OPL client not initialised"
    return _client


def _get_models(approach: str) -> BaseTrajectoryModel:
    if approach not in _models:
        available = ", ".join(_models.keys()) or "(none loaded)"
        raise HTTPException(
            503,
            f"Model for approach '{approach}' not loaded. Available: {available}",
        )
    return _models[approach]


def _load_pretrained_models(pretrained_root: Path) -> None:
    """Load the latest pretrained model for each approach found in the directory."""
    if not pretrained_root.exists():
        return

    approaches = get_all_approaches()

    for approach_name, approach_cls in approaches.items():
        approach_dir = pretrained_root / approach_name
        if not approach_dir.exists():
            continue

        # Find the latest model file in date-sorted subdirectories
        ext = ".pt" if approach_name == "neural_network" else ".joblib"
        candidates = sorted(approach_dir.glob(f"*/model{ext}"))
        if not candidates:
            continue

        try:
            model = approach_cls()
            model.load(candidates[-1])
            _models[approach_name] = model
        except Exception:
            pass  # Skip approaches that fail to load


# ---------------------------------------------------------------------------
# Lifespan — load DB + models once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _models

    db_path_env = os.environ.get("OPL_DB_PATH")
    db_path = Path(db_path_env) if db_path_env else None
    _client = opl.OPL(db_path=db_path)

    model_path_env = os.environ.get("OPL_MODEL_PATH")
    if model_path_env:
        # Legacy single-model path — load as gradient_boosting
        mp = Path(model_path_env)
        if mp.exists():
            from opl.analytics import TrajectoryModel

            model = TrajectoryModel()
            model.load(mp)
            _models["gradient_boosting"] = model
    else:
        # Load all approaches from pretrained directory
        pretrained = Path(os.environ.get("OPL_PRETRAINED_DIR", "pretrained"))
        _load_pretrained_models(pretrained)

        # Fallback: check for legacy flat structure (pretrained/*/model.joblib)
        if not _models and pretrained.exists():
            candidates = sorted(pretrained.glob("*/model.joblib"))
            if candidates:
                from opl.analytics import TrajectoryModel

                model = TrajectoryModel()
                model.load(candidates[-1])
                _models["gradient_boosting"] = model

    yield


app = FastAPI(title="OPL Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class LifterSummary(BaseModel):
    name: str
    competition_count: int
    best_total_kg: float | None


class EntryOut(BaseModel):
    date: datetime.date
    federation: str
    meet_name: str
    equipment: str
    event: str
    bodyweight_kg: float | None
    weight_class_kg: str | None
    best3_squat_kg: float | None
    best3_bench_kg: float | None
    best3_deadlift_kg: float | None
    total_kg: float | None
    place: str
    dots: float | None
    wilks: float | None
    age: float | None
    tested: bool | None


class LifterDetail(BaseModel):
    name: str
    competition_count: int
    best_total_kg: float | None
    best_squat_kg: float | None
    best_bench_kg: float | None
    best_deadlift_kg: float | None
    entries: list[EntryOut]


class PredictRequest(BaseModel):
    lifter_name: str
    target_date: datetime.date | None = None
    target_bodyweight_kg: float | None = None
    approach: str = "gradient_boosting"


class PredictionOut(BaseModel):
    next_total_kg: float | None
    next_squat_kg: float | None
    next_bench_kg: float | None
    next_deadlift_kg: float | None
    confidence_interval: tuple[float, float] | None
    trajectory_curve: list[tuple[int, float]]
    target_date: datetime.date | None
    target_bodyweight_kg: float | None
    approach: str


class ApproachOut(BaseModel):
    name: str
    display_name: str
    description: str
    available: bool


class PercentileRequest(BaseModel):
    lift: str  # "squat", "bench", "deadlift", "total"
    weight_kg: float
    sex: str  # "M" or "F"
    equipment: str  # "Raw", "Wraps", etc.


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/search", response_model=list[LifterSummary])
def search_lifters(q: str = Query(..., min_length=2), limit: int = Query(20, le=50)):
    """Search lifters by name substring."""
    client = _get_client()
    results = client.search_lifters(q, limit=limit)
    return [
        LifterSummary(
            name=lifter.name,
            competition_count=lifter.competition_count,
            best_total_kg=lifter.best_total_kg,
        )
        for lifter in results
    ]


@app.get("/api/lifter/{name:path}", response_model=LifterDetail)
def get_lifter(name: str):
    """Get full lifter details and competition history."""
    client = _get_client()
    lifter = client.lifter(name)
    if lifter is None:
        raise HTTPException(404, f"Lifter '{name}' not found")
    history = lifter.history()
    return LifterDetail(
        name=lifter.name,
        competition_count=lifter.competition_count,
        best_total_kg=lifter.best_total_kg,
        best_squat_kg=lifter.best_squat_kg,
        best_bench_kg=lifter.best_bench_kg,
        best_deadlift_kg=lifter.best_deadlift_kg,
        entries=[
            EntryOut(
                date=e.date,
                federation=e.federation,
                meet_name=e.meet_name,
                equipment=e.equipment.value,
                event=e.event.value,
                bodyweight_kg=e.bodyweight_kg,
                weight_class_kg=e.weight_class_kg,
                best3_squat_kg=e.best3_squat_kg,
                best3_bench_kg=e.best3_bench_kg,
                best3_deadlift_kg=e.best3_deadlift_kg,
                total_kg=e.total_kg,
                place=e.place,
                dots=e.dots,
                wilks=e.wilks,
                age=e.age,
                tested=e.tested,
            )
            for e in history
        ],
    )


@app.get("/api/approaches", response_model=list[ApproachOut])
def get_approaches():
    """List all registered prediction approaches and their availability."""
    approaches = get_all_approaches()
    return [
        ApproachOut(
            name=name,
            display_name=cls.display_name,
            description=cls.description,
            available=name in _models,
        )
        for name, cls in approaches.items()
    ]


@app.post("/api/predict", response_model=PredictionOut)
def predict(req: PredictRequest):
    """Predict a lifter's next performance using a specified approach."""
    client = _get_client()
    model = _get_models(req.approach)

    lifter = client.lifter(req.lifter_name)
    if lifter is None:
        raise HTTPException(404, f"Lifter '{req.lifter_name}' not found")
    if lifter.competition_count < 1:
        raise HTTPException(400, "Lifter has no competition entries")

    try:
        pred = model.predict(
            lifter,
            target_date=req.target_date,
            target_bodyweight_kg=req.target_bodyweight_kg,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc

    return PredictionOut(
        next_total_kg=pred.next_total_kg,
        next_squat_kg=pred.next_squat_kg,
        next_bench_kg=pred.next_bench_kg,
        next_deadlift_kg=pred.next_deadlift_kg,
        confidence_interval=pred.confidence_interval,
        trajectory_curve=pred.trajectory_curve,
        target_date=pred.target_date,
        target_bodyweight_kg=pred.target_bodyweight_kg,
        approach=req.approach,
    )


@app.get("/api/stats")
def stats():
    """Database stats."""
    client = _get_client()
    return client.stats()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "db_loaded": _client is not None,
        "models_loaded": list(_models.keys()),
        "model_count": len(_models),
    }
