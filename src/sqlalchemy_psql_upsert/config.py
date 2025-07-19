"""
config.py - PostgreSQL database configuration module.

This module provides configuration management for PostgreSQL database connections
using environment variables. It supports automatic loading of environment variables
from .env files and provides a convenient way to construct database URIs.

Environment Variables:
    PGSQL_HOST: PostgreSQL server hostname or IP address
    PGSQL_PORT: PostgreSQL server port (default: 5432)
    PGSQL_USER: Database username for authentication
    PGSQL_PASS: Database password for authentication
    PGSQL_NAME: Name of the target database

Example:
    >>> from sqlalchemy_psql_upsert.config import PgConfig
    >>> config = PgConfig()
    >>> engine = create_engine(config.uri())
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class PgConfig:
    """
    PostgreSQL database configuration dataclass.

    This class encapsulates all PostgreSQL connection parameters, automatically
    loading them from environment variables with sensible defaults where applicable.

    Attributes:
        host: PostgreSQL server hostname or IP address
        port: PostgreSQL server port number (default: "5432")
        user: Database username for authentication
        password: Database password for authentication
        dbname: Name of the target database

    Environment Variables:
        PGSQL_HOST: Sets the host attribute
        PGSQL_PORT: Sets the port attribute (default: "5432")
        PGSQL_USER: Sets the user attribute
        PGSQL_PASS: Sets the password attribute
        PGSQL_NAME: Sets the dbname attribute

    Example:
        >>> config = PgConfig()
        >>> print(config.uri())
        'postgresql+psycopg2://user:pass@localhost:5432/mydb'

    Note:
        The port is stored as a string to match the format expected by
        database URI construction and environment variable parsing.
    """
    host: str = os.getenv('PGSQL_HOST', '')
    port: str = os.getenv('PGSQL_PORT', '5432')
    user: str = os.getenv('PGSQL_USER', '')
    password: str = os.getenv('PGSQL_PASS', '')
    dbname: str = os.getenv('PGSQL_NAME', '')

    def uri(self) -> str:
        """
        Generate a PostgreSQL connection URI string.

        Constructs a SQLAlchemy-compatible PostgreSQL connection URI using the
        psycopg2 driver from the configured connection parameters.

        Returns:
            A PostgreSQL connection URI string in the format:
            'postgresql+psycopg2://user:password@host:port/database'

        Example:
            >>> config = PgConfig()
            >>> config.host = "localhost"
            >>> config.user = "myuser"
            >>> config.password = "mypass"
            >>> config.dbname = "mydb"
            >>> config.uri()
            'postgresql+psycopg2://myuser:mypass@localhost:5432/mydb'

        Note:
            This method does not perform any validation of the connection
            parameters. Ensure all required fields (host, user, password, dbname)
            are set before using the returned URI for database connections.
        """
        return f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'

    def __str__(self) -> str:
        """
        Return a string representation of the configuration.

        Returns a formatted string showing the connection details with
        the password masked for security.

        Returns:
            A string representation with masked password.

        Example:
            >>> config = PgConfig()
            >>> str(config)
            'PgConfig(host=localhost, port=5432, user=myuser, password=****, dbname=mydb)'
        """
        masked_password = '****' if self.password else ''
        return (f'PgConfig(host={self.host}, port={self.port}, user={self.user}, '
                f'password={masked_password}, dbname={self.dbname})')
