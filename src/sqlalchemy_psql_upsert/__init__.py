"""
SQLAlchemy PostgreSQL Upsert

A Python library for intelligent PostgreSQL upsert operations with
advanced conflict resolution and multi-threaded processing.
"""

from sqlalchemy import create_engine, text
from .client import PostgresqlUpsert
from .config import PgConfig


def test_connection(config=None, engine=None):
    """Test database connectivity"""
    try:
        test_engine = engine or create_engine(config.uri() if config else PgConfig().uri())
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)


# Public API - what users should import
__all__ = [
    "PostgresqlUpsert",
    "PgConfig",
    "test_connection",
]
