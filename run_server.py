"""
Server Startup Script for Project Meswak (FastAPI + Uvicorn)
"""

import sys
import uvicorn
from app.config import settings

def start():
    print("=" * 70)
    print(f"   STARTING {settings.APP_NAME}")
    print(f"   Serving API & Interactive Dashboard on http://localhost:{settings.PORT}")
    print(f"   API Documentation (Swagger): http://localhost:{settings.PORT}/docs")
    print("=" * 70)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start()

