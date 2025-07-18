"""
main.py - Main script for upserting data to PostgreSQL tables.
"""

import csv
import json
import logging
import os
import pandas as pd
import re
import time

# from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, inspect, text, bindparam
from sqlalchemy import Engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import OperationalError
from tqdm import tqdm
from typing import List, Tuple, Union
from .config import PgConfig

logger = logging.getLogger(__name__)
logger.propagate = True


class PostgresqlUpsert:

    def __init__(self, config: PgConfig = None, engine: Engine = None, debug: bool = False):
        if engine:
            self.engine = engine
        else:
            self.config = config or PgConfig()
            self.engine = create_engine(self.config.uri())

        if debug:
            logger.setLevel(logging.DEBUG)

    def create_engine(self) -> Engine:
        uri = PgConfig().uri()
        return create_engine(uri)

    def list_tables(self):
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def _get_constraints(self, dataframe: pd.DataFrame,
                         table_name: str, schema: str = "public") -> Tuple[List[str], List[List[str]]]:

        if dataframe.empty:
            logger.warning("Received empty DataFrame, skipping _get_constraints.")
            return [], []

        inspector = inspect(self.engine)
        pk_cols = inspector.get_pk_constraint(table_name, schema).get("constrained_columns", [])
        uniques = inspector.get_unique_constraints(table_name, schema)

        unique_sets = [
            u["column_names"]
            for u in uniques
            if all(col in dataframe.columns for col in u["column_names"])
        ]

        if pk_cols and all(c in dataframe.columns for c in pk_cols):
            return pk_cols, [pk_cols] + unique_sets

        return pk_cols, unique_sets

    def _check_conflicts(self, dataframe: pd.DataFrame, constraint: Union[str, List[str]],
                         table_name: str, schema: str = "public") -> pd.Index:
        """
        Return the index (pandas Index) of rows in the DataFrame that would conflict with existing rows in the table.
        Uses SQLAlchemy parameter expansion for IN clauses.
        """

        cols = [constraint] if isinstance(constraint, str) else (constraint or [])
        if not cols:
            logger.warning("No constraint columns provided, skipping _check_conflicts.")
            return pd.Index([])

        inspector = inspect(self.engine)
        table_columns = {col['name'] for col in inspector.get_columns(table_name, schema=schema)}

        df_columns = set(dataframe.columns)
        if not all(col in df_columns and col in table_columns for col in cols):
            logger.warning(
                f"Constraint {cols} columns not present in both DataFrame and table {schema}.{table_name}, skipping _check_conflicts.")
            return pd.Index([])

        df_subset = dataframe[cols]
        if df_subset.empty:
            logger.warning("Empty subset dataframe to check for conflicts, skipping _check_conflicts.")
            return pd.Index([])

        # Sanitize identifiers (basic)
        safe_cols = [f'"{col}"' for col in cols]
        safe_table = f'"{schema}"."{table_name}"'

        if len(cols) == 1:
            # Single-column constraint
            vals = df_subset[cols[0]].tolist()
            sql = text(f"""
                SELECT DISTINCT {safe_cols[0]}
                FROM {safe_table}
                WHERE {safe_cols[0]} IN :vals
            """).bindparams(bindparam('vals', expanding=True))
        else:
            # Multi-column constraint
            vals = [tuple(row) for row in df_subset[cols].to_numpy()]
            col_tuple = f"({', '.join(safe_cols)})"
            sql = text(f"""
                SELECT DISTINCT {', '.join(safe_cols)}
                FROM {safe_table}
                WHERE {col_tuple} IN :vals
            """).bindparams(bindparam('vals', expanding=True))

        with self.engine.connect() as conn:
            result = conn.execute(sql, {'vals': vals})
            existing = set(tuple(row) if len(cols) > 1 else (row[0],) for row in result.fetchall())

        conflict_indices = df_subset[df_subset[cols].apply(tuple, axis=1).isin(existing)].index

        return conflict_indices

    def _remove_multi_conflict_rows(self, dataframe: pd.DataFrame, table_name: str, schema: str = "public",
                                    max_workers: int = 4) -> pd.DataFrame:

        if dataframe.empty:
            logger.warning("Received empty Dataframe, skipping multi-conflict removal.")
            return dataframe

        _, uniques = self._get_constraints(dataframe, table_name, schema)
        if not uniques:
            logger.warning(f"No unique constraints found for table {table_name}, skipping multi-conflict removal.")
            return dataframe

        conflict_counts = pd.Series(0, index=dataframe.index)

        def worker(constraint):
            try:
                return self._check_conflicts(dataframe,  constraint, table_name, schema)
            except Exception as e:
                logger.warning(f"Constraint {constraint} check failed: {e}")
                return pd.Index([])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, c): c for c in uniques}
            for future in as_completed(futures):
                try:
                    conflict_idx = future.result()
                    # Increment count for each index
                    conflict_counts.loc[conflict_idx] += 1
                except Exception as e:
                    logger.error(f"Thread failed: {e}")

        to_remove = conflict_counts[conflict_counts > 1].index

        removed_df = dataframe.loc[to_remove]

        if not removed_df.empty:
            logger.warning(f"{len(removed_df)} rows removed due to multiple unique constraint conflicts")
            logger.debug(f"Removed rows:\n{removed_df}")

        return dataframe.drop(index=to_remove)

    def _do_upsert(self, chunk: pd.DataFrame, table: Table, pk_cols: List[str], valid_constraints: List[List[str]]):
        insert_stmt = insert(table).values(chunk.to_dict(orient="records"))
        update_cols = {c.name: insert_stmt.excluded[c.name] for c in table.columns if c.name not in pk_cols}

        stmt = None
        for constraint in valid_constraints:
            if all(col in chunk.columns for col in constraint):
                stmt = insert_stmt.on_conflict_do_update(
                    index_elements=constraint,
                    set_=update_cols
                )
                break

        if stmt is None:
            logger.warning("[FALLBACK] Inserted chunk with no matching constraint.")
            stmt = insert_stmt

        retries = 0
        max_retries, backoff_factor = 3, 2

        while retries <= max_retries:
            try:
                with self.engine.begin() as conn:
                    conn.execute(stmt)
                return
            except OperationalError as e:
                retries += 1
                logger.warning(f"Transient DB error on chunk, retry {retries}/{max_retries}: {e}")
                time.sleep(backoff_factor ** retries)
            except Exception as e:
                logger.error(f"Permanent error on chunk: {e}")
                return

        logger.error("Max retries reached, giving up on chunk")

    def upsert_dataframe(self, dataframe: pd.DataFrame, table_name: str, schema: str = "public",
                         chunk_size: int = 10_000, max_workers: int = 4, deduplicate: bool = True):
        if dataframe.empty:
            logger.warning("Received empty DataFrame. Skipping upsert.")
            return True

        df = dataframe.copy()

        metadata = MetaData(schema=schema)
        try:
            table = Table(table_name, metadata, autoload_with=self.engine)
        except Exception as e:
            logger.error(f"Destination table '{schema}.{table_name}' not found: {e}")
            return False

        pk_cols, valid_constraints = self._get_constraints(df, table, schema)

        if not valid_constraints:
            logger.warning(
                f"No PK or UNIQUE constraints found on '{schema}.{table_name}'. Performing plain INSERT.")
        else:
            df = self._conflict_aware_deduplicate(df, valid_constraints)

        if deduplicate and valid_constraints:
            logger.debug(f"Running conflict-aware deduplication with constraints: {valid_constraints}")
            df = self._conflict_aware_deduplicate(df, valid_constraints)
        else:
            logger.warning("Deduplication skipped; relying on PostgreSQL ON CONFLICT for conflict resolution.")

        logger.info(f"Upserting {len(df)} rows into {schema}.{table_name}...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                futures.append(executor.submit(
                    self._do_upsert, chunk, table, pk_cols, valid_constraints
                ))

            with tqdm(total=len(futures), desc="Upserting chunks") as pbar:
                for future in as_completed(futures):
                    if exc := future.exception():
                        logger.error(f"Chunk failed: {exc}")
                    pbar.update(1)

        logger.info("Upsert complete.")
        return True

    def dump_table_to_csv(self, table_name: str, schema: str = "public",
                          output_dir: str = "data", chunksize: int = 100_000):
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{table_name}.csv")

        chunk_iter = pd.read_sql_table(table_name, schema, self.engine, chunksize=chunksize)

        first_chunk = True
        for chunk in tqdm(chunk_iter, desc=f"Dumping {table_name}"):
            mode = 'w' if first_chunk else 'a'
            header = first_chunk

            chunk = chunk.applymap(
                lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x
            )
            chunk.to_csv(csv_path, mode=mode, index=False, header=header, encoding='utf-8',
                         quoting=csv.QUOTE_ALL, quotechar='"', escapechar='\\', lineterminator='\n'
                         )

    def export_pgsql_dtypes(self, table_name, schema="public", output_dir="./"):
        dtypes_path = os.path.join(output_dir, f"{table_name}.dtypes.json")

        insp = inspect(self.engine)
        columns = insp.get_columns(table_name, schema=schema)

        def map_pgsql_to_pd(dtype_str):
            dtype_str = dtype_str.upper()

            if re.search(r'\b(INT|INT2|INT4|INT8|SMALLINT|BIGINT)\b', dtype_str):
                return "int64"
            elif re.search(r'\b(FLOAT|FLOAT4|FLOAT8|REAL|DOUBLE PRECISION|NUMERIC|DECIMAL)\b', dtype_str):
                return "float64"
            elif re.search(r'\b(BOOL|BOOLEAN)\b', dtype_str):
                return "bool"
            elif re.search(r'\b(DATE|TIMESTAMP|TIME|TIMESTAMPTZ)\b', dtype_str):
                return "datetime64[ns]"
            else:
                return "string"

        dtypes = {
            col["name"]: map_pgsql_to_pd(str(col["type"]))
            for col in columns
        }

        with open(dtypes_path, "w") as f:
            json.dump(dtypes, f, indent=2)

# from cleantext import clean
# from urllib.parse import quote_plus

# class MSSQLConnector:
#     def __init__(self) -> None:
#         db_host: str = os.getenv("MSSQL_HOST")
#         db_port: str = os.getenv("MSSQL_PORT", "1433")
#         db_user: str = os.getenv("MSSQL_USER")
#         db_pass: str = os.getenv("MSSQL_PASS")
#         db_name: str = os.getenv("MSSQL_DATABASE")
#         connection_string = (
#             f"DRIVER={{ODBC Driver 17 for SQL Server}};"
#             f"SERVER={db_host},{db_port};"
#             f"DATABASE={db_name};"
#             f"UID={db_user};"
#             f"PWD={db_pass};"
#         )
#         params = quote_plus(connection_string)
#         uri = f"mssql+pyodbc:///?odbc_connect={params}"
#         self.engine = self.create_engine(uri)

#     def create_engine(self, uri) -> Engine:
#         try:
#             engine = create_engine(uri)
#             return engine
#         except Exception as e:
#             logger.error(f"Failed to create engine: {e}")

#     def list_tables(self):
#         inspector = inspect(self.engine)
#         return inspector.get_table_names()

#     def get_table_schema(self, table_name: str) -> dict:
#         logger.info(f"Querying schema for table '{table_name}'")

#         try:
#             with self.engine.connect() as conn:
#                 query = f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'"
#                 result = conn.execute(query)
#                 schema = {row[0]: row[1] for row in result}
#                 return schema

#         except Exception as err:
#             logger.error(f"Error querying MSSQL table: {err}")
#             return None

#     def dump_table_to_csv(self, table_name: str, output_dir: str = "data", chunksize: int = 100_000):
#         os.makedirs(output_dir, exist_ok=True)
#         csv_path = os.path.join(output_dir, f"{table_name}.csv")
#         dtypes_path = os.path.join(output_dir, f"{table_name}.dtypes.json")

#         chunk_iter = pd.read_sql_table(table_name, self.engine, chunksize=chunksize)

#         first_chunk = True
#         for chunk in tqdm(chunk_iter, desc=f"Dumping {table_name}"):
#             mode = 'w' if first_chunk else 'a'
#             header = first_chunk

#             chunk = chunk.applymap(
#                 lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x
#             )
#             chunk.to_csv(csv_path, mode=mode, index=False, header=header, encoding='utf-8',
#                          quoting=csv.QUOTE_ALL, quotechar='"', escapechar='\\', lineterminator='\n'
#                          )

#             if first_chunk:
#                 json.dump({col: str(dt) for col, dt in chunk.dtypes.items()}, open(dtypes_path, 'w'))
#                 first_chunk = False

#     def append_df_bulk(self, table_name: str, dataframe: pd.DataFrame, batch_size: int = 1_000) -> bool:
#         logger.info(f"Appending dataframe to table '{table_name}' in bulk")

#         df = dataframe.copy()
#         df.columns = map(str, df.columns)

#         for column in df.select_dtypes(include=['object']):
#             df[column] = df[column].apply(lambda x: clean(x, fix_unicode=True, lower=False) if x is not None else None)

#         try:
#             total_batches = (len(df) + batch_size - 1) // batch_size

#             with tqdm(total=total_batches, desc=f"Appending to {table_name}") as pbar:
#                 for i in range(total_batches):
#                     start_idx = i * batch_size
#                     end_idx = min((i + 1) * batch_size, len(df))
#                     batch = df.iloc[start_idx:end_idx]

#                     with self.engine.begin() as conn:
#                         batch.to_sql(
#                             name=table_name,
#                             con=conn,
#                             if_exists='append',
#                             index=False,
#                             method='multi'
#                         )
#                     pbar.update(1)

#             logger.info(f"Successfully appended {len(df)} rows to table {table_name}")
#             return True

#         except Exception as e:
#             logger.error(f"Failed to append dataframe to table: {e}")
#             raise


# def main():

#     # UPSERTING DATA FROM CSV FILES TO PGSQL

#     pgsql = PSQLConnector()
#     last_crawl_csv_files = [f for f in os.listdir("./winescraper/files") if "last_crawl" in f and f.endswith(".csv")]
#     store_dict = [
#         {"store_name": "bancadoramon", "table_name": "B2C_CONC_BANCADORAMON"},
#         {"store_name": "divinho", "table_name": "B2C_CONC_DIVINHO"},
#         {"store_name": "encontrevinhos", "table_name": "B2C_CONC_ENCONTRE_VINHOS"},
#         {"store_name": "evino", "table_name": "B2C_CONC_EVINO"},
#         {"store_name": "grandcru", "table_name": "B2C_CONC_GRANDCRU"},
#         {"store_name": "mistral", "table_name": "B2C_CONC_MISTRAL"},
#         {"store_name": "paodeacucar", "table_name": "B2C_CONC_PAO_ACUCAR"},
#         {"store_name": "stmarche", "table_name": "B2C_CONC_SAINT_MARCHE"},
#         {"store_name": "vinci", "table_name": "B2C_CONC_VINCI"},
#         {"store_name": "worldwine", "table_name": "B2C_CONC_WORLD_WINE"}
#     ]

#     for item in store_dict:
#         store = item["store_name"]
#         table = item["table_name"]

#         csv_file = next((file for file in last_crawl_csv_files if store.lower()
#                         in file.lower() and 'last_crawl' in file.lower()), None)

#         if not csv_file:
#             logger.error(f"No CSV file found for store: {store}")
#             continue

#         try:
#             logger.info(f"Upserting: {csv_file}")
#             csv_path = f"./winescraper/files/{csv_file}"

#             pgsql.export_pgsql_dtypes(table_name=table, schema="Holos_FB", output_dir="./data")

#             dtypes_path = f"./data/{table}.dtypes.json"
#             dtypes = json.load(open(dtypes_path))

#             date_cols = [col for col, dt in dtypes.items() if dt.startswith("datetime") and col != "insert_time"]

#             df = pd.read_csv(csv_path, encoding='utf-8', dtype=dtypes,
#                              parse_dates=date_cols, quotechar='"', engine='python')

#             pgsql.upsert_dataframe(df, table_name=table, schema="Holos_FB")

#             logger.info(f"Successfully upserted: {csv_file}")

#         except Exception as e:
#             logger.error(f"Error upserting: {csv_file}: {e}")


# if __name__ == "__main__":
#     main()
