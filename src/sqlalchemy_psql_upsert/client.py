"""
client.py - PostgreSQL upsert client implementation with temporary table approach.

This module provides PostgreSQL upsert functionality with support for:
- Temporary table + JOIN based conflict resolution
- Comprehensive constraint handling (PK, UNIQUE)
- Automatic NaN to NULL conversion for pandas DataFrames
- Efficient bulk operations using SQLAlchemy
- Progress tracking with tqdm
"""

import logging
import pandas as pd
import uuid
from sqlalchemy import create_engine, inspect, text, insert
from sqlalchemy import Engine, MetaData, Table, UniqueConstraint
from tqdm import tqdm
from typing import List, Tuple, Optional
from .config import PgConfig

logger = logging.getLogger(__name__)
logger.propagate = True


class PostgresqlUpsert:
    """
    PostgreSQL upsert utility class with support for conflict resolution and multi-threading.

    This class provides methods to upsert pandas DataFrames into PostgreSQL tables with
    automatic handling of primary key and unique constraint conflicts using a temporary
    table approach. All pandas NaN values (np.nan, pd.NaType, None) are automatically
    converted to PostgreSQL NULL values.
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
            PermissionError: If database user lacks CREATE TEMP TABLE privileges.
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

        # Verify temporary table creation privileges
        self._verify_temp_table_privileges()

    def create_engine(self) -> Engine:
        """Create a new SQLAlchemy engine using default configuration."""
        uri = PgConfig().uri()
        logger.debug(f"Creating new database engine with URI: {uri}")
        return create_engine(uri)

    def _list_tables(self) -> List[str]:
        """
        Get a list of all table names in the database.

        Returns:
            List of table names in the database.
        """
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        logger.debug(f"Found {len(tables)} tables in database: {tables}")
        return tables

    def _verify_temp_table_privileges(self) -> None:
        """
        Verify that the database user has privileges to create temporary tables.

        Note: The test temp table is not explicitly dropped as it will be automatically
        cleaned up when the session ends. This avoids issues where users have CREATE
        privileges but not DROP privileges.

        Raises:
            PermissionError: If the user lacks CREATE TEMP TABLE privileges.
        """
        test_temp_table = f"test_temp_privileges_{uuid.uuid4().hex[:8]}"

        try:
            with self.engine.begin() as conn:
                # Try to create a minimal test temporary table
                conn.execute(text(f'CREATE TEMP TABLE "{test_temp_table}" (test_col INTEGER)'))
                # Note: Not explicitly dropping - PostgreSQL will auto-cleanup on session end

            logger.debug("Temporary table creation privileges verified successfully")

        except Exception as e:
            error_msg = (
                f"Database user lacks CREATE TEMP TABLE privileges. "
                f"Error: {str(e)}. "
                f"Please ensure your PostgreSQL user has TEMPORARY privilege on the database, "
                f"or grant it with: GRANT TEMPORARY ON DATABASE your_database TO your_user;"
            )
            logger.error(error_msg)
            raise PermissionError(error_msg) from e

    def _get_constraints(self, table_name: str, schema: str = "public") -> Tuple[List[str], List[List[str]]]:
        """
        Get primary key and unique constraints for a table.

        Args:
            table_name: Name of the target table
            schema: Database schema name

        Returns:
            Tuple of (primary_key_columns, list_of_unique_constraint_columns)
        """
        try:
            metadata = MetaData()
            target_table = Table(table_name, metadata, autoload_with=self.engine, schema=schema)

            pk_cols = [col.name for col in target_table.primary_key.columns]

            uniques = []
            for constraint in target_table.constraints:
                if isinstance(constraint, UniqueConstraint):
                    constraint_cols = [col.name for col in constraint.columns]
                    uniques.append(constraint_cols)

            # Sort unique constraints by first column name for consistent ordering
            uniques.sort(key=lambda x: x[0])

            if not pk_cols and not uniques:
                logger.warning(f"No PK or UNIQUE constraints found on '{schema}.{table_name}'.")
                return [], []

            logger.debug(f"Retrieved constraints for {schema}.{table_name}: PK={pk_cols}, Uniques={uniques}")

            return (pk_cols, uniques)

        except Exception as e:
            logger.error(f"Failed to retrieve constraints for table {schema}.{table_name}: {str(e)}")
            raise

    def _get_dataframe_constraints(self, dataframe: pd.DataFrame, table_name: str,
                                   schema: str = "public") -> Tuple[List[str], List[List[str]]]:
        """
        Get constraints that are applicable to the given DataFrame.
        Only returns constraints where ALL required columns are present in DataFrame.
        """
        if dataframe.empty:
            logger.warning("Received empty DataFrame for constraint analysis, returning empty constraints")
            return [], []

        try:
            pk_cols, uniques = self._get_constraints(table_name, schema)

            # Check PK: only include if ALL PK columns are present
            filtered_pk = pk_cols if pk_cols and all(col in dataframe.columns for col in pk_cols) else []

            if pk_cols and not filtered_pk:
                missing_pk_cols = [col for col in pk_cols if col not in dataframe.columns]
                logger.debug(f"Primary key columns {missing_pk_cols} not found in DataFrame columns")

            # Check unique constraints: only include if ALL constraint columns are present
            filtered_uniques = []
            for constraint_cols in uniques:
                if all(col in dataframe.columns for col in constraint_cols):
                    filtered_uniques.append(constraint_cols)
                else:
                    missing_cols = [col for col in constraint_cols if col not in dataframe.columns]
                    logger.debug(f"Skipping unique constraint {constraint_cols} - missing columns: {missing_cols}")

            logger.debug(f"Found {len(filtered_uniques)} usable unique constraints for upsert operations")
            return filtered_pk, filtered_uniques

        except Exception as e:
            logger.error(f"Failed to retrieve constraints for dataframe: {str(e)}")
            raise

    def _convert_nan_to_none(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Convert pandas DataFrame NaN values to None.

        This method handles all pandas NaN variants including:
        - numpy.nan (np.nan)
        - pandas._libs.missing.NAType (pd.NA)
        - Python None
        - float('nan')

        Args:
            dataframe: DataFrame to convert

        Returns:
            DataFrame with NaN values converted to None (which becomes NULL in PostgreSQL)
        """
        if dataframe.empty or dataframe is None:
            logger.warning("Received empty DataFrame. Skipping NaN conversion.")
            return dataframe

        try:
            df = dataframe.copy()

            # Replace all NaN variants with None
            for col in df.columns:
                df[col] = df[col].where(pd.notna(df[col]), None)

            logger.debug(f"Converted NaN values to None for {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Failed to convert NaN values to None: {e}")
            raise

    def _create_temp_table(self, dataframe: pd.DataFrame, target_table_name: str,
                           schema: str = "public") -> str:
        """
        Create a temporary table with the same structure as the target table.

        Args:
            dataframe: DataFrame to determine column types
            target_table_name: Name of the target table to copy structure from
            schema: Database schema name

        Returns:
            Name of the created temporary table
        """
        # Generate unique temp table name
        temp_table_name = f"temp_upsert_{uuid.uuid4().hex[:8]}"

        try:
            # Get target table structure
            metadata = MetaData()
            target_table = Table(target_table_name, metadata, autoload_with=self.engine, schema=schema)

            # Create CREATE TABLE statement for temp table
            columns_def = []
            for col in target_table.columns:
                if col.name in dataframe.columns:
                    col_type = str(col.type.compile(dialect=self.engine.dialect))
                    columns_def.append(f'"{col.name}" {col_type}')

            columns_str = ", ".join(columns_def)

            create_temp_sql = f"""
                CREATE TEMP TABLE "{temp_table_name}" (
                    {columns_str}
                )
            """

            with self.engine.begin() as conn:
                conn.execute(text(create_temp_sql))

            logger.debug(f"Created temporary table: {temp_table_name}")
            return temp_table_name

        except Exception as e:
            logger.error(f"Failed to create temporary table: {e}")
            raise

    def _insert_into_temp_table(self, dataframe: pd.DataFrame, temp_table_name: str) -> Tuple[bool, int]:
        """
        Insert all DataFrame data into the temporary table using SQLAlchemy.

        Args:
            dataframe: DataFrame to insert
            temp_table_name: Name of the temporary table

        Returns:
            Tuple of (success: bool, affected_rows: int)

        Raises:
            Exception: If insertion fails for any reason
        """
        try:
            # Convert NaN to None
            df_clean = self._convert_nan_to_none(dataframe)

            # Convert DataFrame to list of dictionaries
            records = df_clean.to_dict('records')

            if not records:
                logger.debug("No records to insert into temporary table")
                return True, 0

            # Get temp table metadata for SQLAlchemy insert
            metadata = MetaData()
            temp_table = Table(temp_table_name, metadata, autoload_with=self.engine)

            # Use SQLAlchemy's insert construct for bulk insert
            insert_stmt = insert(temp_table).values(records)

            with self.engine.begin() as conn:
                result = conn.execute(insert_stmt)
                affected_rows = result.rowcount

            logger.debug(f"INSERT temporary table operation completed successfully, affected {affected_rows} rows")

            return True, affected_rows

        except Exception as e:
            logger.error(f"Failed to insert data into temporary table {temp_table_name}: {e}")
            raise

    def _resolve_conflicts_with_temp_table(self, temp_table_name: str, target_table_name: str,
                                           schema: str = "public") -> Tuple[bool, int]:
        """
        Resolve conflicts between temporary table and target table using advanced MERGE with CTEs.

        This method handles sophisticated upsert logic with comprehensive conflict resolution:
        1. Analyze all conflicts across all constraints simultaneously
        2. Filter out rows that conflict with multiple target rows (ambiguous)
        3. Deduplicate temp rows that target the same existing row
        4. Use PostgreSQL MERGE for atomic upsert operation
        5. Track and log all skipped rows with reasons

        Args:
            temp_table_name: Name of the temporary table
            target_table_name: Name of the target table
            schema: Database schema name (default: "public")

        Returns:
            Tuple of (success: bool, affected_rows: int)

        Raises:
            ValueError: If no constraints are found for conflict resolution
            Exception: If MERGE operation fails for any reason
        """
        try:
            # Get table metadata and constraints
            metadata = MetaData()
            target_table = Table(target_table_name, metadata, autoload_with=self.engine, schema=schema)
            all_columns = [col.name for col in target_table.columns]

            pk_cols, uniques = self._get_constraints(target_table_name, schema)

            if not pk_cols and not uniques:
                raise ValueError("No constraints found - No conflict resolution to perform")

            # Build all constraints list for dynamic processing
            all_constraints = [pk_cols] if pk_cols else []
            all_constraints.extend(uniques)

            # Build constraint join conditions dynamically
            constraint_conditions = []
            for constraint_cols in all_constraints:
                if len(constraint_cols) == 1:
                    # Single column constraint: t.col = target.col
                    col = constraint_cols[0]
                    constraint_conditions.append(f'temp."{col}" = target."{col}"')
                else:
                    # Multi-column constraint: (t.col1 = target.col1 AND t.col2 = target.col2)
                    multi_conditions = []
                    for col in constraint_cols:
                        multi_conditions.append(f'temp."{col}" = target."{col}"')
                    constraint_conditions.append(f"({' AND '.join(multi_conditions)})")

            # Combine all constraint conditions with OR
            join_on_clause = " OR ".join(constraint_conditions)

            # Build GROUP BY clause for all temp table columns
            group_by_columns = ", ".join([f'temp."{col}"' for col in all_columns])
            group_by_ctid_and_columns = "temp.ctid, " + group_by_columns

            # Build ORDER BY clause for deterministic row ranking
            order_by_columns = ", ".join([f'"{col}"' for col in all_columns])  # noqa

            # Build SELECT clause for clean rows
            select_columns = ", ".join([f'"{col}"' for col in all_columns])

            # Build UPDATE SET clause (exclude PK columns to avoid conflicts)
            update_columns = [col for col in all_columns if col not in (pk_cols or [])]
            if update_columns:
                update_set_clause = ", ".join([f'"{col}" = src."{col}"' for col in update_columns])
            else:
                # If all columns are PK, create a dummy update
                update_set_clause = f'"{all_columns[0]}" = src."{all_columns[0]}"'

            # Build VALUES clause for INSERT
            values_clause = ", ".join([f'src."{col}"' for col in all_columns])

            # Determine partition column for deduplication (prefer PK, fallback to first column)
            partition_col = pk_cols[0] if pk_cols else all_columns[0]

            # Build comprehensive MERGE SQL with dynamic constraints
            merge_sql = f"""
                WITH temp_with_conflicts AS (
                    SELECT temp.*, temp.ctid,
                           COUNT(DISTINCT target."{partition_col}") AS conflict_targets,
                           MAX(target."{partition_col}") AS conflicted_target_id
                    FROM "{temp_table_name}" temp
                    LEFT JOIN "{schema}"."{target_table_name}" target
                      ON {join_on_clause}
                    GROUP BY {group_by_ctid_and_columns}
                ),

                -- Filter rows conflicting with more than one target row
                filtered AS (
                    SELECT * FROM temp_with_conflicts
                    WHERE conflict_targets <= 1
                ),

                -- Deduplicate by keeping the last row per conflicted target
                ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY COALESCE(conflicted_target_id, "{partition_col}")
                               ORDER BY ctid DESC
                           ) AS row_rank
                    FROM filtered
                ),

                -- Final rows to upsert
                clean_rows AS (
                    SELECT {select_columns}
                    FROM ranked
                    WHERE row_rank = 1
                )

                -- Execute the MERGE operation
                MERGE INTO "{schema}"."{target_table_name}" AS tgt
                USING clean_rows AS src
                ON ({join_on_clause.replace('temp.', 'tgt.').replace('target.', 'src.')})
                WHEN MATCHED THEN
                    UPDATE SET {update_set_clause}
                WHEN NOT MATCHED THEN
                    INSERT ({select_columns})
                    VALUES ({values_clause})
            """
            logger.debug(f"Generated MERGE SQL: {merge_sql}")

            with self.engine.begin() as conn:
                # Execute the MERGE operation
                result = conn.execute(text(merge_sql))
                affected_rows = result.rowcount
                logger.debug(f"MERGE operation completed successfully, affected {affected_rows} rows")

            return True, affected_rows

        except Exception as e:
            logger.error(f"Failed to execute MERGE operation: {e}")
            raise

    def _cleanup_temp_table(self, temp_table_name: str, schema: str = "public") -> None:
        """
        Clean up the temporary table.

        Args:
            temp_table_name: Name of the temporary table to drop
            schema: Database schema name (default: "public")

        Raises:
            Exception: If cleanup operation fails (logged but not re-raised)
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{schema}.{temp_table_name}"'))

            logger.debug(f"Cleaned up temporary table: {schema}.{temp_table_name}")

        except Exception as e:
            logger.error(f"Failed to cleanup temporary table {schema}.{temp_table_name}: {e}")

    def upsert_dataframe(self, dataframe: pd.DataFrame, table_name: str, schema: str = "public") -> Tuple[bool, int]:
        """
        Upsert a pandas DataFrame into a PostgreSQL table using temporary table approach.

        This method automatically handles:
        - Multiple constraint conflicts using temporary table + JOIN approach
        - NaN value conversion: All pandas NaN values are converted to PostgreSQL NULL
        - Efficient bulk operations without chunking

        Args:
            dataframe: Input DataFrame to upsert
            table_name: Target table name in the database
            schema: Database schema name (default: "public")

        Returns:
            Tuple of (success: bool, affected_rows: int) - success is always True if no exception

        Raises:
            ValueError: If table doesn't exist or other validation errors
            OperationalError: For database connection or transient errors
            Exception: For any database errors during processing
        """
        if dataframe.empty or dataframe is None:
            logger.warning("Received empty DataFrame. Skipping upsert.")
            return True, 0

        logger.info(f"Starting upsert operation for {len(dataframe)} rows into {schema}.{table_name}")

        # Validate target table exists
        inspector = inspect(self.engine)
        if table_name not in inspector.get_table_names(schema=schema):
            error_msg = f"Destination table '{schema}.{table_name}' not found."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get constraints
        pk_cols, uniques = self._get_constraints(table_name, schema)

        if not pk_cols and not uniques:
            logger.info("No PK or UNIQUE constraints found, loading data using INSERT with no conflict resolution.")

            df_clean = self._convert_nan_to_none(dataframe)
            records = df_clean.to_dict('records')

            if not records:
                logger.warning("No records to insert after NaN conversion")
                return True, 0

            metadata = MetaData()
            target_table = Table(table_name, metadata, autoload_with=self.engine, schema=schema)
            insert_stmt = insert(target_table).values(records)

            with self.engine.begin() as conn:
                result = conn.execute(insert_stmt)
                affected_rows = result.rowcount
            logger.info(f"INSERT operation completed successfully, affected {affected_rows} rows")

            return True, affected_rows

        logger.info(f"Found constraints: PK={pk_cols}, Uniques={uniques}")

        temp_table_name = None

        try:
            with tqdm(total=6, desc="Processing upsert", unit="step") as pbar:
                # Step 1: Create temporary table
                temp_table_name = self._create_temp_table(dataframe, table_name, schema)
                pbar.update(1)
                pbar.set_description("Created temporary table")

                # Step 2: Insert data into temp table
                self._insert_into_temp_table(dataframe, temp_table_name)
                pbar.update(1)
                pbar.set_description("Inserted data to temp table")

                # Step 3: Resolve conflicts using SQL
                _, affected_rows = self._resolve_conflicts_with_temp_table(temp_table_name, table_name, schema)
                pbar.update(4)
                pbar.set_description("Completed successfully")

            return True, affected_rows

        except Exception as e:
            logger.error(f"Upsert operation failed: {e}")
            raise

        finally:
            # Always cleanup temp table if it exists
            if temp_table_name:
                self._cleanup_temp_table(temp_table_name)
