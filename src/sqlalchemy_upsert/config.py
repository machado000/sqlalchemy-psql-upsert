from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PgConfig:
    host: str = os.getenv("PGSQL_HOST", "")
    port: str = os.getenv("PGSQL_PORT", "5432")
    user: str = os.getenv("PGSQL_USER", "")
    password: str = os.getenv("PGSQL_PASS", "")
    dbname: str = os.getenv("PGSQL_NAME", "")

    def uri(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
