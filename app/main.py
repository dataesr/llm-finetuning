from fastapi import FastAPI
from app.version import __version__
from app.api import router as api_router

app = FastAPI(version=__version__)

# include api router
app.include_router(api_router)
