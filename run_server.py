"""
Server Startup Script for PureAir® (FastAPI + Uvicorn)
"""

import os
import sys
import uvicorn
from app.config import settings

def start():
    port = int(os.environ.get("PORT", settings.PORT))
    print("=" * 70)
    print(f"   STARTING {settings.APP_NAME}")
    print(f"   Serving API & Interactive Dashboard on http://{settings.HOST}:{port}")
    print(f"   API Documentation (Swagger): http://{settings.HOST}:{port}/docs")
    print("=" * 70)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start()

