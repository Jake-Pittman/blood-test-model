# api.py
from __future__ import annotations
import os
from typing import Dict, Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from recommender import NutriRecommender

MODEL_DIR = os.getenv("MODEL_DIR", "models/FinalModel")
FOODS_PATH = os.getenv("FOODS_PATH", "data/processed/foods_dictionary_plus_enriched.parquet")

app = FastAPI(title="Nutri Recommender API", version="1.0")

# CORS for bolt.new (add your app origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    labs: Dict[str, float] = Field(
        default_factory=dict,
        description="Either human keys (ldl, hdl, triglycerides, fasting_glucose, ...) or NHANES codes (LBDLDL, LBDHDL, …)",
    )
    top_k: int = 50
    strict_mode: bool = True
    apply_filters: bool = True

class RecommendResponseItem(BaseModel):
    category: Optional[str] = None
    desc: Optional[str] = None
    _score: float

class RecommendResponse(BaseModel):
    results: List[RecommendResponseItem]

@app.on_event("startup")
def _load_artifacts():
    global REC, FOODS, SHOW_COLS
    REC = NutriRecommender(model_dir=MODEL_DIR)
    # Load foods once; Parquet preferred. Fallback to CSV transparently.
    if FOODS_PATH.lower().endswith(".parquet"):
        FOODS = pd.read_parquet(FOODS_PATH)
    else:
        FOODS = pd.read_csv(FOODS_PATH)
    SHOW_COLS = [c for c in ["category", "desc", "_score"] if c in FOODS.columns]

@app.get("/health")
def health():
    return {"ok": True, "model_dir": MODEL_DIR, "foods_rows": int(len(FOODS))}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    df = REC.recommend(
        foods=FOODS,
        labs=body.labs,
        top_k=body.top_k,
        strict_mode=body.strict_mode,
        apply_filters=body.apply_filters,
        return_debug=False,
    )
    cols = [c for c in ["category", "desc", "_score"] if c in df.columns]
    return {"results": [dict(row[cols]) for _, row in df.iterrows()]}

