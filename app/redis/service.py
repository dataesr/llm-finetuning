import os
import redis
from rq import Queue, Worker
from app.logging import get_logger

REDIS_URL = "redis://redis:6379/0"
QUEUE_NAME = "llm"

logger = get_logger(__name__)


def get_queue() -> Queue:
    """Get redis queue

    Returns:
        Queue: redis queue
    """
    queue = Queue("llm", connection=redis.from_url(REDIS_URL), default_timeout=216000)
    return queue


def run_worker():
    """Run redis worker"""
    if os.getenv("ENABLE_REDIS_WORKER", "false").lower() == "true":
        logger.debug("Start redis worker")
        worker = Worker("llm", connection=redis.from_url(REDIS_URL))
        worker.work()
