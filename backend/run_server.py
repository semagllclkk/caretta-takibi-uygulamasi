#!/usr/bin/env python3
"""
Simple server runner script
"""
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Starting Caretta Track API server...")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Disable reload to avoid import issues
        log_level="info",
    )
