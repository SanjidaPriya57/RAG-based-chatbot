import os
import redis.asyncio as redis
import json
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Global Redis client
redis_client: Optional[redis.Redis] = None


async def init_cache():
    """Initialize Redis connection."""
    global redis_client

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    try:
        redis_client = await redis.from_url(redis_url, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis cache initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis cache: {e}")
        # Continue without cache if Redis is unavailable
        redis_client = None


async def get_cache(key: str) -> Optional[Any]:
    """Get value from cache."""
    if redis_client is None:
        return None

    try:
        value = await redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return None


async def set_cache(key: str, value: Any, ttl: int = 3600) -> bool:
    """Set value in cache with TTL."""
    if redis_client is None:
        return False

    try:
        await redis_client.setex(key, ttl, json.dumps(value))
        return True
    except Exception as e:
        logger.error(f"Cache set error: {e}")
        return False


async def delete_cache(key: str) -> bool:
    """Delete value from cache."""
    if redis_client is None:
        return False

    try:
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Cache delete error: {e}")
        return False


async def close_cache():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis cache closed")
