from fastapi import FastAPI
from app.version import __version__
from app.redis import service as redis_service
from app.api import router as api_router

# Create fastapi application
app = FastAPI(version=__version__)

# Run redis worker
redis_service.run_worker()

# Include api router
app.include_router(api_router)
