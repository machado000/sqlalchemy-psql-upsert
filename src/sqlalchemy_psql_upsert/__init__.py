"""
SQLAlchemy PostgreSQL Upsert

A Python library for intelligent PostgreSQL upsert operations with
advanced conflict resolution and multi-threaded processing.
"""

import logging
from sqlalchemy import create_engine, text, Engine
from typing import Optional, Tuple
from .client import PostgresqlUpsert
from .config import PgConfig

# Package logger
logger = logging.getLogger(__name__)
logger.propagate = True


def setup_logging(level: int = logging.INFO,
                  format_string: Optional[str] = None) -> None:
    """
    Setup logging configuration.

    Args:
        level (int): Logging level (default: INFO)
        format_string (Optional[str]): Custom format string
    """
    if format_string is None:
        format_string = '%(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
        ]
    )


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
        logger.debug("Testing database connection")
        test_engine = engine or create_engine(config.uri() if config else PgConfig().uri())
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection test successful")
        return True, "Connection successful"

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Database connection test failed: {error_msg}")
        return False, error_msg


# Public API - what users should import
__all__ = [
    "PostgresqlUpsert",
    "PgConfig",
    "setup_logging",
    "test_connection",
]
