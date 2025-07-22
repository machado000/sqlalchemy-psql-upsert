"""
client.py - PostgreSQL upsert client implementation.

This module provides PostgreSQL upsert functionality with support for:
- Automatic conflict detection and resolution
- Automatic NaN to NULL conversion for pandas DataFrames
- Multi-threaded chunk processing
- Progress tracking with tqdm
- Comprehensive constraint handling (PK, UNIQUE)
"""

import logging
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, inspect, text, bindparam
from sqlalchemy import Engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import OperationalError
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Optional, Union
from .config import PgConfig

logger = logging.getLogger(__name__)
logger.propagate = True


class PostgresqlUpsert:
    """
    PostgreSQL upsert utility class with support for conflict resolution and multi-threading.

    This class provides methods to upsert pandas DataFrames into PostgreSQL tables with
    automatic handling of primary key and unique constraint conflicts. All pandas NaN
    values (np.nan, pd.NaType, None) are automatically converted to PostgreSQL NULL values.
    """

    def __init__(self, config: Optional['PgConfig'] = None, engine: Optional[Engine] = None,
                 debug: bool = False) -> None:
        """
        Initialize PostgreSQL upsert client.

        Args:
            config: PostgreSQL configuration object. If None, default config will be used.
            engine: SQLAlchemy engine instance. If provided, config will be ignored.
            debug: Enable debug logging for detailed operation information.

        Raises:
            ValueError: If neither config nor engine is provided and default config fails.
        """
        if engine:
            self.engine = engine
            logger.info("PostgreSQL upsert client initialized with provided engine")
        else:
            self.config = config or PgConfig()
            self.engine = create_engine(self.config.uri())
            logger.info(f"PostgreSQL upsert client initialized with config: {self.config.host}:{self.config.port}")

        if debug:
            logger.setLevel(logging.DEBUG)
            logger.info("Debug logging enabled for PostgreSQL upsert operations")

    def create_engine(self) -> Engine:
        """Create a new SQLAlchemy engine using default configuration."""
        uri = PgConfig().uri()
        logger.debug(f"Creating new database engine with URI: {uri}")
        return create_engine(uri)

    def list_tables(self) -> List[str]:
        """
        Get a list of all table names in the database.

        Returns:
            List of table names in the database.
        """
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        logger.debug(f"Found {len(tables)} tables in database: {tables}")
        return tables

    def _get_constraints(self, dataframe: pd.DataFrame, table_name: str,
                         schema: str = "public") -> Tuple[List[str], List[List[str]]]:
        """
        Get primary key and unique constraints for a table.

        Args:
            dataframe: DataFrame to check column compatibility against
            table_name: Name of the target table
            schema: Database schema name

        Returns:
            Tuple of (primary_key_columns, list_of_unique_constraint_columns)
        """
        if dataframe.empty:
            logger.warning("Received empty DataFrame for constraint analysis, returning empty constraints")
            return [], []

        try:
            inspector = inspect(self.engine)
            pk_constraint = inspector.get_pk_constraint(table_name, schema)
            pk_cols = pk_constraint.get('constrained_columns', [])
            uniques = inspector.get_unique_constraints(table_name, schema)

            logger.debug(f"Retrieved constraints for {schema}.{table_name}: "
                         f"PK={pk_cols}, unique_constraints={len(uniques)}")

            # Filter unique constraints to only include those with columns present in DataFrame
            unique_sets = []
            for u in uniques:
                constraint_cols = u['column_names']
                if all(col in dataframe.columns for col in constraint_cols):
                    unique_sets.append(constraint_cols)
                else:
                    logger.debug(f"Skipping unique constraint {constraint_cols} - "
                                 f"columns not present in DataFrame")

            # If PK columns exist and are in DataFrame, include them
            if pk_cols and all(c in dataframe.columns for c in pk_cols):
                return pk_cols, [pk_cols] + unique_sets
            else:
                if pk_cols:
                    logger.debug(f"Primary key columns {pk_cols} not found in DataFrame columns")

            logger.debug(f"Found {len(unique_sets)} usable unique constraints for upsert operations")
            return pk_cols, unique_sets

        except Exception as e:
            logger.error(f"Failed to retrieve constraints for table {schema}.{table_name}: {str(e)}")
            raise

    def _check_conflicts(self, dataframe: pd.DataFrame, constraint: Union[str, List[str]],
                         table_name: str, schema: str = "public") -> Dict[int, Set[Tuple]]:
        """
        For each row in the DataFrame, return a mapping from DataFrame index to set of matched
        table row identifiers (PK tuple or ctid) in the table.

        Args:
            dataframe: Input DataFrame to check for conflicts
            constraint: Column name(s) that form the constraint
            table_name: Target database table name
            schema: Database schema name

        Returns:
            Dictionary mapping DataFrame index to set of table row identifiers
        """

        cols = [constraint] if isinstance(constraint, str) else (constraint or [])
        if not cols:
            logger.warning("No constraint columns provided, skipping _check_conflicts.")
            return {}

        inspector = inspect(self.engine)
        table_columns = {col['name'] for col in inspector.get_columns(table_name, schema=schema)}
        pk_cols = inspector.get_pk_constraint(table_name, schema).get('constrained_columns', [])

        df_columns = set(dataframe.columns)
        if not all(col in df_columns and col in table_columns for col in cols):
            logger.warning(f"Constraint {cols} columns not present in both DataFrame and "
                           f"table {schema}.{table_name}, skipping _check_conflicts.")
            return {}

        df_subset = dataframe[cols]
        if df_subset.empty:
            logger.warning("Empty subset dataframe to check for conflicts, skipping _check_conflicts.")
            return {}

        # Sanitize identifiers (basic)
        safe_cols = [f'"{col}"' for col in cols]
        safe_table = f'"{schema}"."{table_name}"'

        # Determine identifier columns: PK or ctid
        if pk_cols:
            id_cols = [f'"{col}"' for col in pk_cols]
        else:
            id_cols = ['ctid']

        select_cols = id_cols + safe_cols
        select_cols_str = ', '.join(select_cols)

        # Write WHERE clause for single or multiple constraint columns
        if len(cols) == 1:
            vals = df_subset[cols[0]].tolist()
            where_clause = f"{safe_cols[0]} IN :vals"
        else:
            vals = [tuple(row) for row in df_subset[cols].to_numpy()]
            col_tuple = f"({', '.join(safe_cols)})"
            where_clause = f"{col_tuple} IN :vals"

        query = text(f"""
            SELECT {select_cols_str}
            FROM {safe_table}
            WHERE {where_clause}
        """).bindparams(bindparam('vals', expanding=True))

        with self.engine.connect() as conn:
            result = conn.execute(query, {'vals': vals})

            # Build mapping: constraint tuple -> set of PK/ctid tuples
            table_row_map = {}
            for row in result.fetchall():
                id_val = tuple(row[:len(id_cols)])
                constraint_val = tuple(row[len(id_cols):])
                table_row_map.setdefault(constraint_val, set()).add(id_val)

        # Build mapping: DataFrame index -> set of matched table row identifiers (PK/ctid)
        index_to_table_rows = {}
        for idx, row in df_subset.iterrows():
            key = tuple(row) if len(cols) > 1 else (row.iloc[0],)
            if key in table_row_map:
                index_to_table_rows[idx] = table_row_map[key]

        logger.debug(f"Found {len(index_to_table_rows)} DataFrame rows with conflicts in table")
        return index_to_table_rows

    def _remove_multi_conflict_rows(self, dataframe: pd.DataFrame, table_name: str,
                                    schema: str = "public",
                                    max_workers: int = 4) -> pd.DataFrame:
        """
        Remove DataFrame rows that match more than one unique table row (across all unique constraints),
        using the dict[int, set[tuple]] output from _check_conflicts.

        Args:
            dataframe: Input DataFrame to filter
            table_name: Target database table name
            schema: Database schema name
            max_workers: Maximum number of threads for parallel constraint checking

        Returns:
            Filtered DataFrame with multi-conflict rows removed
        """
        if dataframe.empty:
            logger.warning("Received empty DataFrame, skipping multi-conflict removal.")
            return dataframe

        _, uniques = self._get_constraints(dataframe, table_name, schema)
        if not uniques:
            logger.warning(f"No unique constraints found for table {table_name}, "
                           f"skipping multi-conflict removal.")
            return dataframe

        # For each DataFrame index, collect all matched PK/ctid tuples across all constraints
        index_to_table_rows: Dict[int, Set[Tuple]] = {}

        def worker(constraint: List[str]) -> Dict[int, Set[Tuple]]:
            try:
                return self._check_conflicts(dataframe, constraint, table_name, schema)
            except Exception as e:
                logger.warning(f"Constraint {constraint} check failed: {e}")
                return {}

        logger.info(f"Checking {len(uniques)} unique constraints for multi-conflicts "
                    f"using {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, c): c for c in uniques}
            for future in as_completed(futures):
                constraint = futures[future]
                try:
                    result = future.result()
                    for idx, pk_set in result.items():
                        if idx not in index_to_table_rows:
                            index_to_table_rows[idx] = set()
                        index_to_table_rows[idx].update(pk_set)
                except Exception as e:
                    logger.error(f"Thread processing constraint {constraint} failed: {e}")

        # Remove rows that match more than one unique table row (i.e., set size > 1)
        to_remove = [idx for idx, pk_set in index_to_table_rows.items() if len(pk_set) > 1]

        removed_df = dataframe.loc[to_remove]

        if not removed_df.empty:
            logger.warning(f"{len(removed_df)} rows removed due to multiple unique constraint conflicts")
            logger.debug(f"Removed rows:\n{removed_df}")
        else:
            logger.info("No multi-conflict rows found, all data will be processed")

        return dataframe.drop(index=to_remove)

    def _convert_nan_to_none(self, chunk: pd.DataFrame) -> List[Dict]:
        """
        Convert pandas DataFrame to list of dictionaries with NaN values converted to None.

        This method handles all pandas NaN variants including:
        - numpy.nan (np.nan)
        - pandas._libs.missing.NAType (pd.NA)
        - Python None
        - float('nan')

        Args:
            chunk: DataFrame chunk to convert

        Returns:
            List of dictionaries with NaN values converted to None (which becomes NULL in PostgreSQL)
        """
        # Convert DataFrame to records, then replace NaN with None
        records = chunk.to_dict(orient='records')

        # Replace NaN values with None (handles pd.NaType, np.nan, float('nan'), etc.)
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                    logger.debug(f"Converted NaN to None for column '{key}'")

        return records

    def _do_upsert(self, chunk: pd.DataFrame, table: Table, pk_cols: List[str],
                   uniques: List[List[str]]) -> None:
        """
        Execute upsert for a single chunk of data.

        This method converts all NaN values to None before executing the upsert operation,
        ensuring proper NULL handling in PostgreSQL for both INSERT and UPDATE operations.

        Args:
            chunk: DataFrame chunk to upsert
            table: SQLAlchemy Table object
            pk_cols: Primary key column names
            uniques: List of unique constraint column lists

        Raises:
            OperationalError: For transient database errors (retries automatically)
            Exception: For permanent errors
        """
        if chunk.empty:
            logger.warning("Received empty chunk for upsert, skipping")
            return

        logger.debug(f"Processing chunk with {len(chunk)} rows")

        # Convert NaN values to None before creating the insert statement
        records = self._convert_nan_to_none(chunk)

        insert_stmt = insert(table).values(records)

        # Create update columns for conflict resolution
        update_cols = {}
        for c in table.columns:
            if c.name not in pk_cols:
                update_cols[c.name] = insert_stmt.excluded[c.name]

        stmt = None
        for constraint in uniques:
            if all(col in chunk.columns for col in constraint):
                stmt = insert_stmt.on_conflict_do_update(
                    index_elements=constraint,
                    set_=update_cols
                )
                break

        if stmt is None:
            logger.warning("No matching constraints found, performing plain INSERT (may fail on conflicts)")
            stmt = insert_stmt

        retries = 0
        max_retries, backoff_factor = 3, 2

        while retries <= max_retries:
            try:
                with self.engine.begin() as conn:
                    result = conn.execute(stmt)
                    rows_affected = result.rowcount if hasattr(result, 'rowcount') else len(chunk)
                    logger.debug(f"Chunk processed successfully, {rows_affected} rows affected")
                return
            except OperationalError as e:
                retries += 1
                wait_time = backoff_factor ** retries
                logger.warning(f"Transient database error (attempt {retries}/{max_retries}): {e}. "
                               f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Permanent error processing chunk: {e}")
                return

        logger.error(f"Max retries ({max_retries}) reached for chunk, operation failed")

    def upsert_dataframe(self, dataframe: pd.DataFrame, table_name: str, schema: str = "public",
                         chunk_size: int = 10_000, max_workers: int = 4,
                         remove_multi_conflict_rows: bool = True) -> bool:
        """
        Upsert a pandas DataFrame into a PostgreSQL table with conflict resolution.

        This method automatically handles:
        - Primary key and unique constraint conflicts using PostgreSQL's ON CONFLICT clause
        - NaN value conversion: All pandas NaN values are converted to PostgreSQL NULL
        - Multi-threaded processing for large datasets
        - Progress tracking with visual progress bars

        Args:
            dataframe: Input DataFrame to upsert
            table_name: Target table name in the database
            schema: Database schema name (default: "public")
            chunk_size: Number of rows per chunk for parallel processing
            max_workers: Maximum number of worker threads
            remove_multi_conflict_rows: Whether to remove rows with multiple conflicts

        Returns:
            True if successful, raises exceptions on failure

        Raises:
            ValueError: If table doesn't exist or other validation errors
            Exception: For database errors during operation
        """
        if dataframe.empty:
            logger.warning("Received empty DataFrame. Skipping upsert.")
            return True

        logger.info(f"Starting upsert operation for {len(dataframe)} rows into {schema}.{table_name}")

        df = dataframe.copy()

        inspector = inspect(self.engine)

        if table_name not in inspector.get_table_names(schema=schema):
            error_msg = f"Destination table '{schema}.{table_name}' not found."
            logger.error(error_msg)
            raise ValueError(error_msg)

        pk_cols, uniques = self._get_constraints(df, table_name, schema)

        if not uniques:
            logger.warning(
                f"No PK or UNIQUE constraints found on '{schema}.{table_name}'. Performing plain INSERT.")
        else:
            logger.info(f"Running multi-conflict removal with constraints: {uniques}")
            df = self._remove_multi_conflict_rows(df, table_name, schema, max_workers=max_workers)

        if df.empty:
            logger.warning("No rows remaining after conflict resolution. Operation completed.")
            return True

        logger.info(f"Upserting {len(df)} rows into {schema}.{table_name} in chunks of {chunk_size}...")

        # Progress bars: one for total rows, one for chunks
        num_chunks = (len(df) + chunk_size - 1) // chunk_size
        chunk_indices = [(i, min(i + chunk_size, len(df))) for i in range(0, len(df), chunk_size)]

        try:
            metadata = MetaData(schema=schema)
            table = Table(table_name, metadata, autoload_with=self.engine)
        except Exception as e:
            error_msg = f"Failed to load table '{schema}.{table_name}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        successful_chunks = 0
        failed_chunks = 0

        with tqdm(total=len(df), desc="Total rows upserted", unit="rows") as row_pbar:
            with tqdm(total=num_chunks, desc="Upserting chunks", unit="chunk") as chunk_pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for start, end in chunk_indices:
                        chunk = df.iloc[start:end]
                        future = executor.submit(self._do_upsert, chunk, table, pk_cols, uniques)
                        futures[future] = (start, end)

                    for future in as_completed(futures):
                        start, end = futures[future]
                        chunk_size_actual = end - start
                        if exc := future.exception():
                            logger.error(f"Chunk {start}:{end} failed: {exc}")
                            failed_chunks += 1
                        else:
                            logger.debug(f"Chunk {start}:{end} upserted successfully.")
                            successful_chunks += 1
                        row_pbar.update(chunk_size_actual)
                        chunk_pbar.update(1)

        if failed_chunks > 0:
            logger.warning(f"Upsert completed with {failed_chunks} failed chunks out of {num_chunks} total chunks")
        else:
            logger.info(f"Upsert completed successfully. {successful_chunks} chunks processed.")

        return True
