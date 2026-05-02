"""
backend/main.py  — FastAPI uygulama giriş noktası (test amaçlı minimal sürüm)
"""
from __future__ import annotations

import logging
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Uygulama
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Caretta Track API",
    description="Caretta Caretta kaplumbağa yüz tanıma sistemi",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS Middleware (Frontend'in API'ye erişebilmesi için)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",      # Frontend dev sunucusu
        "http://127.0.0.1:8080",
        "http://localhost:3000",      # Alternative port
        "http://127.0.0.1:3000",
        "*",                          # Production: specific origins
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API Router (SOLID: Presentation Layer)
# ---------------------------------------------------------------------------
from api.endpoints import router as api_router
app.include_router(api_router)

# ---------------------------------------------------------------------------
# Lazy singletons — ilk istekte başlatılır (torch yükleme süresi için)
# Artık api/endpoints.py'de tanımlanıyor, burası eski versiyonu silindi
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Endpoint'ler
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def root():
    """Ana sayfa — API sağlığını kontrol eder."""
    return {
        "status": "ok",
        "message": "Caretta Track API çalışıyor.",
        "api_docs": "/docs",
        "endpoints": {
            "health": "GET /api/health",
            "train": "POST /api/train",
            "predict": "POST /api/predict",
        }
    }


@app.get("/health", tags=["health"])
def health():
    """FastAPI tarafından otomatik sağlık kontrolü."""
    return {"status": "healthy", "version": app.version}
