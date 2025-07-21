"""
SQLAlchemy PostgreSQL Upsert

A Python library for intelligent PostgreSQL upsert operations with
advanced conflict resolution and multi-threaded processing.
"""

from typing import Optional, Tuple
from sqlalchemy import create_engine, text, Engine
from .client import PostgresqlUpsert
from .config import PgConfig


def test_connection(config: Optional[PgConfig] = None, engine: Optional[Engine] = None) -> Tuple[bool, str]:
    """
    Test database connectivity.

    Args:
        config: PostgreSQL configuration object. If None, default config will be used.
        engine: SQLAlchemy engine instance. If provided, config will be ignored.

    Returns:
        Tuple of (success: bool, message: str)

    Example:
        >>> success, message = test_connection()
        >>> if success:
        ...     print("Database connection OK")
        ... else:
        ...     print(f"Connection failed: {message}")
    """
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
