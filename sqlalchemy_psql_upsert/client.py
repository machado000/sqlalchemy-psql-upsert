"""
PostgreSQL upsert client implementation with temporary table approach.

This module provides PostgreSQL upsert functionality with support for:
- Temporary table + JOIN based conflict resolution
- Comprehensive constraint handling (Primary Key, UNIQUE constraints)
- Comprehensive NaN to NULL conversion for pandas DataFrames (handles pandas NaN,
  string representations like "nan", "null", "none", and empty strings)
- Efficient bulk operations using SQLAlchemy
- Progress tracking with tqdm
- Detailed skip reason tracking for debugging

The main class `PostgresqlUpsert` implements a dual temporary table approach:
1. Raw temp table: Stores all input data with skip_reason tracking
2. Clean temp table: Contains conflict-resolved data ready for merge

This approach handles complex scenarios including:
- Multiple unique constraints on the same table
- Rows that conflict with multiple target records
- Duplicate rows in the source data
- Missing constraint columns in the DataFrame

Type hints are provided for all public and private methods following
PEP 484 standards.
"""

import logging
import pandas as pd
import uuid

from sqlalchemy import create_engine, inspect, text, insert
from sqlalchemy import Engine, MetaData, Table, UniqueConstraint, Column, Text
from tqdm import tqdm
from typing import Optional, Any
from .config import PgConfig

logger = logging.getLogger(__name__)
logger.propagate = True


class PostgresqlUpsert:
    """
    PostgreSQL upsert utility class with support for conflict resolution.

    This class provides methods to upsert pandas DataFrames into PostgreSQL tables with
    automatic handling of primary key and unique constraint conflicts using a temporary
    table approach. Comprehensive NaN value conversion: All pandas NaN values (np.nan,
    pd.NaType, None) and string representations ("nan", "null", "none", empty strings)
    are automatically converted to PostgreSQL NULL values.
    """

    def __init__(self, config: Optional[PgConfig] = None, engine: Optional[Engine] = None,
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
        """
        Create a new SQLAlchemy engine using default configuration.

        Returns:
            SQLAlchemy Engine instance configured with default PostgreSQL settings.
        """
        uri = PgConfig().uri()
        logger.debug(f"Creating new database engine with URI: {uri}")
        return create_engine(uri)

    def upsert_dataframe(self, dataframe: pd.DataFrame, table_name: str, schema: str = "public",
                         return_skipped: bool = False) -> int | tuple[int, pd.DataFrame]:
        """
        Upsert a pandas DataFrame into a PostgreSQL table using dual temporary table approach.

        This method automatically handles:
        - Multiple constraint conflicts using temporary table approach
        - Comprehensive NaN value conversion: All pandas NaN values and string representations
          ("nan", "null", "none", empty strings) are converted to PostgreSQL NULL
        - Efficient bulk operations with conflict resolution
        - Optional return of skipped rows for analysis

        Args:
            dataframe: Input DataFrame to upsert
            table_name: Target table name in the database
            schema: Database schema name (default: "public")
            return_skipped: If True, returns DataFrame with skipped rows (default: False)

        Returns:
            int: Number of affected rows in the target table
            tuple[int, pd.DataFrame]: When return_skipped=True, returns (affected_rows, skipped_rows_df)

        Raises:
            ValueError: If table doesn't exist or other validation errors
            Exception: For database connection, transient, or processing errors
        """
        skipped_df = pd.DataFrame()

        if dataframe.empty or dataframe is None:
            logger.warning("Received empty DataFrame. Skipping upsert.")
            return (0, skipped_df) if return_skipped else 0

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

            affected_rows = self._batch_insert_dataframe(dataframe, table_name, schema)

            return (affected_rows, skipped_df) if return_skipped else affected_rows

        logger.info(f"Found constraints: PK={pk_cols}, Uniques={uniques}")

        raw_table_name = None
        clean_table_name = None

        try:
            with tqdm(total=6, desc="Processing upsert", unit="step") as pbar:
                # Step 1: Create both temporary tables
                raw_table_name, clean_table_name = self._create_temp_tables(dataframe, table_name, schema)
                pbar.update(1)
                pbar.set_description(f'{"Created temporary tables":>25}')

                # Step 2: Insert data into raw temp table
                self._batch_insert_dataframe(dataframe, raw_table_name, schema=None)
                pbar.update(1)
                pbar.set_description(f'{"Loaded raw data":>25}')

                # Step 3: Populate clean temp table with conflict resolution
                self._solve_constraint_conflicts(raw_table_name, table_name, schema)
                pbar.update(1)
                pbar.set_description(f'{"Resolved conflicts":>25}')

                # Step 4: Populate clean temp table with conflict resolution
                clean_rows_count = self._populate_clean_temp_table(raw_table_name, clean_table_name)
                pbar.update(1)
                pbar.set_description(f'{"Populated clean table":>25}')

                # Step 5: Get skipped rows if requested
                if return_skipped:
                    skipped_df = self._get_skipped_rows(raw_table_name) or pd.DataFrame()
                    pbar.update(1)
                    pbar.set_description(f'{"Fetch skipped rows":>25}')
                else:
                    pbar.update(1)

                # Step 6: Execute MERGE operation
                affected_rows = self._merge_from_clean_temp_table(clean_table_name, table_name, schema)
                pbar.update(1)
                pbar.set_description(f'{"Executed MERGE":>25}')

                pbar.update(1)
                pbar.set_description(f'{"Completed successfully":>25}')

            logger.info(f"Upsert completed: {affected_rows} rows affected, "
                        f"{len(dataframe) - clean_rows_count} rows skipped")

            return (affected_rows, skipped_df) if return_skipped else affected_rows

        except Exception as e:
            logger.error(f"Upsert operation failed: {e}")
            raise

        finally:
            # Always cleanup temp tables if they exist
            if raw_table_name or clean_table_name:
                self._cleanup_temp_tables(raw_table_name, clean_table_name)

    def _list_tables(self) -> list[str]:
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

    def _get_constraints(self, table_name: str, schema: str = "public") -> tuple[list[str], list[list[str]]]:
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
                                   schema: str = "public") -> tuple[list[str], list[list[str]]]:
        """
        Get constraints that are applicable to the given DataFrame.
        Only returns constraints where ALL required columns are present in DataFrame.

        Args:
            dataframe: Input DataFrame to analyze for constraint compatibility
            table_name: Name of the target table
            schema: Database schema name (default: "public")

        Returns:
            Tuple of (primary_key_columns, list_of_unique_constraint_columns)
            where each constraint list only includes constraints with all columns
            present in the DataFrame.

        Raises:
            Exception: If constraint retrieval from the database fails
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

    def _create_temp_tables(self, dataframe: pd.DataFrame, target_table_name: str,
                            schema: str = "public") -> tuple[str, str]:
        """
        Create two TEMPORARY tables: one for raw data to upsert and one for clean rows after conflict resolution.

        Args:
            dataframe: DataFrame to determine column types
            target_table_name: Name of the target table to copy structure from
            schema: Database schema name

        Returns:
            Tuple of (raw_temp_table_name, clean_temp_table_name)
        """
        # Generate unique temp table names
        base_uuid = uuid.uuid4().hex[:8]
        raw_table_name = f"temp_raw_{base_uuid}"
        clean_table_name = f"temp_clean_{base_uuid}"

        try:
            # Get target table structure
            source_metadata = MetaData()
            target_table = Table(target_table_name, source_metadata, autoload_with=self.engine, schema=schema)

            # Create new metadata for temp tables
            temp_metadata = MetaData()

            # Create new columns for temp tables (don't reuse existing Column objects)
            raw_columns = []
            clean_columns = []

            for col in target_table.columns:
                if col.name in dataframe.columns:
                    # Create new Column objects with same type but without constraints
                    raw_columns.append(Column(col.name, col.type, nullable=True))
                    clean_columns.append(Column(col.name, col.type, nullable=True))

            # Add skip_reason column only to raw table
            skip_reason_col = Column('skip_reason', Text, nullable=True)
            raw_columns.append(skip_reason_col)

            # Create temp table definitions
            raw_table = Table(raw_table_name, temp_metadata, *raw_columns, prefixes=['TEMPORARY'])
            clean_table = Table(clean_table_name, temp_metadata, *clean_columns, prefixes=['TEMPORARY'])

            # Create both tables
            temp_metadata.create_all(self.engine, tables=[raw_table, clean_table])

            logger.debug(f"Created temporary tables: {raw_table_name} (raw), {clean_table_name} (clean)")
            return raw_table_name, clean_table_name

        except Exception as e:
            logger.error(f"Failed to create temporary tables: {e}")
            raise

    def _is_nan_value(self, value: Any) -> bool:
        """
        Comprehensive NaN detection for various data types.

        This function detects:
        - pandas NaN, np.nan, None
        - String representations: "nan", "NaN", "null", "NULL", "None"
        - Empty strings and whitespace-only strings

        Args:
            value: Any value to check for NaN-like properties

        Returns:
            bool: True if value should be converted to SQL NULL, False otherwise
        """
        # Handle None and pandas/numpy NaN
        if value is None or pd.isna(value):
            return True

        # Handle string representations of NaN/null
        if isinstance(value, str):
            cleaned_value = value.strip().lower()
            if cleaned_value in ('nan', 'null', 'none', ''):
                return True

        return False

    def _batch_insert_dataframe(self, dataframe: pd.DataFrame, table_name: str, schema: Optional[str] = "public",
                                batch_size: int = 5000) -> int:
        """
        Insert all DataFrame data into a table using SQLAlchemy with batched processing.

        Args:
            dataframe: DataFrame to insert
            table_name: Name of the table
            schema: Database schema name (default: "public"). Pass None for temporary tables.
            batch_size: Number of rows to process per batch (default: 5000)

        Returns:
            Number of rows that were successfully inserted

        Raises:
            Exception: If insertion fails for any reason
        """
        if dataframe.empty:
            logger.debug("No records to insert into temporary table")
            return 0

        try:
            # Get temp table metadata for SQLAlchemy insert
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.engine, schema=schema)

            total_rows = len(dataframe)
            total_affected_rows = 0

            # Use single connection for all batches to reduce overhead
            with self.engine.begin() as conn:
                with tqdm(total=total_rows, desc=f'{"Inserting raw data":>25}', unit="rows") as pbar:
                    for start_idx in range(0, total_rows, batch_size):
                        end_idx = min(start_idx + batch_size, total_rows)
                        batch_df = dataframe.iloc[start_idx:end_idx]

                        # Convert DataFrame batch to records with comprehensive NaN to None conversion
                        batch_records = [
                            {k: (None if self._is_nan_value(v) else v) for k, v in row.items()}
                            for row in batch_df.to_dict('records')
                        ]

                        # Execute batch insert
                        insert_stmt = insert(table).values(batch_records)
                        result = conn.execute(insert_stmt)
                        batch_affected = result.rowcount
                        total_affected_rows += batch_affected

                        # Update progress
                        pbar.update(len(batch_records))
                        pbar.set_postfix({
                            'batch': f"{start_idx//batch_size + 1}",
                            'inserted': f"{total_affected_rows}"
                        })

            logger.debug(f"INSERT on table '{table_name}' completed successfully, affected {total_affected_rows} rows")

            return total_affected_rows

        except Exception as e:
            logger.error(f"Failed to insert data into temporary table {table_name}: {e}")
            raise

    def _solve_constraint_conflicts(self, raw_table_name: str, target_table_name: str, schema: str = "public") -> None:
        """
        Resolve constraint conflicts by updating skip_reason column in raw table.
        Uses optimized step-by-step approach for better performance on large datasets.

        Args:
            raw_table_name: Name of the raw data temporary table
            target_table_name: Name of the target table
            schema: Database schema name
        """
        try:
            # Get table metadata and constraints
            metadata = MetaData()
            target_table = Table(target_table_name, metadata, autoload_with=self.engine, schema=schema)
            raw_table = Table(raw_table_name, metadata, autoload_with=self.engine, schema=None)

            # Get columns that exist in both target and raw tables
            target_columns = {col.name for col in target_table.columns}
            raw_columns = {col.name for col in raw_table.columns if col.name != 'skip_reason'}
            common_columns = list(target_columns.intersection(raw_columns))

            pk_cols, uniques = self._get_constraints(target_table_name, schema)

            # Filter constraints to only include those with ALL columns present in raw table
            usable_pk_cols = pk_cols if pk_cols and all(col in raw_columns for col in pk_cols) else []
            usable_uniques = [uc for uc in uniques if all(col in raw_columns for col in uc)]

            if not usable_pk_cols and not usable_uniques:
                # No usable constraints, just mark all rows as accepted
                update_sql = f'UPDATE "{raw_table_name}" SET skip_reason = NULL'

                with self.engine.begin() as conn:
                    conn.execute(text(update_sql))

                logger.debug(f"No constraints found, marked all rows as accepted in '{raw_table_name}'")
                return

            # Build constraint logic for conflict resolution
            usable_constraints = [usable_pk_cols] if usable_pk_cols else []
            usable_constraints.extend(usable_uniques)

            # Build constraint join conditions dynamically
            constraint_conditions = []
            for constraint_cols in usable_constraints:
                if len(constraint_cols) == 1:
                    col = constraint_cols[0]
                    constraint_conditions.append(f'raw."{col}" = target."{col}"')
                else:
                    multi_conditions = [f'raw."{col}" = target."{col}"' for col in constraint_cols]
                    constraint_conditions.append(f"({' AND '.join(multi_conditions)})")

            join_on_clause = " OR ".join(constraint_conditions)
            select_raw_columns = ", ".join([f'raw."{col}"' for col in common_columns])

            with self.engine.begin() as conn:
                # Step 1: Create temporary conflict analysis table for better performance
                conflicts_table = f"temp_conflicts_{uuid.uuid4().hex[:8]}"

                create_conflicts_sql = f"""
                    CREATE TEMP TABLE "{conflicts_table}" AS
                    SELECT raw.ctid as raw_ctid,
                           {select_raw_columns},
                           COUNT(DISTINCT target.ctid) AS conflict_count,
                           MIN(target.ctid) AS target_ctid
                    FROM "{raw_table_name}" raw
                    LEFT JOIN "{schema}"."{target_table_name}" target ON {join_on_clause}
                    GROUP BY raw.ctid, {select_raw_columns}
                """

                conn.execute(text(create_conflicts_sql))

                # Step 2: Mark rows with multiple target conflicts
                mark_multiple_conflicts_sql = f"""
                    UPDATE "{raw_table_name}"
                    SET skip_reason = 'multiple_target_conflicts'
                    WHERE ctid IN (
                        SELECT raw_ctid FROM "{conflicts_table}"
                        WHERE conflict_count > 1
                    )
                """

                conn.execute(text(mark_multiple_conflicts_sql))

                # Step 3: Handle duplicate source rows (keep only the last one per target)
                mark_duplicates_sql = f"""
                    UPDATE "{raw_table_name}"
                    SET skip_reason = 'duplicate_source_rows'
                    WHERE ctid IN (
                        SELECT c1.raw_ctid
                        FROM "{conflicts_table}" c1
                        JOIN "{conflicts_table}" c2 ON (
                            c1.target_ctid = c2.target_ctid
                            AND c1.raw_ctid < c2.raw_ctid
                        )
                        WHERE c1.conflict_count <= 1 AND c2.conflict_count <= 1
                    )
                """

                conn.execute(text(mark_duplicates_sql))

                # Step 4: Mark remaining rows as accepted (skip_reason = NULL)
                mark_accepted_sql = f"""
                    UPDATE "{raw_table_name}"
                    SET skip_reason = NULL
                    WHERE skip_reason IS NULL
                """

                conn.execute(text(mark_accepted_sql))

                # Clean up temporary conflicts table
                conn.execute(text(f'DROP TABLE IF EXISTS "{conflicts_table}"'))

            logger.debug(f"Solved conflicts using optimized approach for '{raw_table_name}'")

        except Exception as e:
            logger.error(f"Failed to solve conflicts: {e}")
            raise

    def _populate_clean_temp_table(self, raw_table_name: str, clean_table_name: str,) -> int:
        """
        Populate the temporary `clean_table` with conflict-resolved rows from
        temporary `raw_table` where skip_reason is NULL.

        Args:
            raw_table_name: Name of the raw data temporary table
            clean_table_name: Name of the clean rows temporary table

        Returns:
            Number of rows inserted into clean temp table
        """
        try:
            # Get table metadata and constraints
            metadata = MetaData()
            raw_table = Table(raw_table_name, metadata, autoload_with=self.engine, schema=None)
            clean_table = Table(clean_table_name, metadata, autoload_with=self.engine, schema=None)  # noqa

            # Get columns that exist in both tables
            raw_columns = {col.name for col in raw_table.columns if col.name != 'skip_reason'}
            clean_columns = {col.name for col in clean_table.columns}
            common_columns = list(raw_columns.intersection(clean_columns))
            select_common_columns = ", ".join([f'"{col}"' for col in common_columns])

            # Now insert clean rows
            insert_clean_rows_sql = f"""
                INSERT INTO "{clean_table_name}" ({select_common_columns})
                SELECT {select_common_columns}
                FROM "{raw_table_name}"
                WHERE skip_reason IS NULL
            """

            with self.engine.begin() as conn:
                result = conn.execute(text(insert_clean_rows_sql))
                clean_rows_count = result.rowcount

            logger.debug(f"Populated clean temp table '{clean_table_name}' with {clean_rows_count} rows")
            return clean_rows_count

        except Exception as e:
            logger.error(f"Failed to populate clean temp table: {e}")
            raise

    def _get_skipped_rows(self, raw_table_name: str) -> Optional[pd.DataFrame]:
        """
        Get rows that were skipped during conflict resolution with detailed skip reasons.

        Args:
            raw_table_name: Name of the raw data temporary table (with skip_reason column)

        Returns:
            DataFrame with skipped rows and their specific skip reasons.
            Returns empty DataFrame if no rows were skipped.
            Returns None if an error occurs during retrieval.
        """
        try:
            # Simply select all skipped rows with their reasons
            skipped_sql = f"""
                SELECT * FROM "{raw_table_name}"
                WHERE skip_reason IS NOT NULL
            """

            with self.engine.begin() as conn:
                result = conn.execute(text(skipped_sql))
                rows = result.fetchall()

            if not rows:
                # Return empty DataFrame with proper column structure
                metadata = MetaData()
                raw_table = Table(raw_table_name, metadata, autoload_with=self.engine)
                columns = [col.name for col in raw_table.columns]
                return pd.DataFrame(columns=columns)

            # Convert to DataFrame with column names from result
            skipped_df = pd.DataFrame(rows, columns=list(result.keys()))

            logger.debug(f"Found {len(skipped_df)} skipped rows with detailed reasons")
            return skipped_df

        except Exception as e:
            logger.error(f"Failed to retrieve skipped rows: {e}")
            return None

    def _merge_from_clean_temp_table(self, clean_table_name: str, target_table_name: str,
                                     schema: str = "public") -> int:
        """
        Execute MERGE operation from clean temp table to target table.

        Args:
            clean_table_name: Name of the clean rows temporary table
            target_table_name: Name of the target table
            schema: Database schema name

        Returns:
            Number of affected rows
        """
        try:
            # Get table metadata and constraints
            metadata = MetaData()
            target_table = Table(target_table_name, metadata, autoload_with=self.engine, schema=schema)
            clean_table = Table(clean_table_name, metadata, autoload_with=self.engine)

            # Get columns that exist in both target and clean tables
            target_columns = {col.name for col in target_table.columns}
            clean_columns = {col.name for col in clean_table.columns}
            common_columns = list(target_columns.intersection(clean_columns))
            select_common_columns = ", ".join([f'"{col}"' for col in common_columns])

            pk_cols, uniques = self._get_constraints(target_table_name, schema)

            # Filter constraints to only include those with ALL columns present in clean table
            usable_pk_cols = pk_cols if pk_cols and all(col in common_columns for col in pk_cols) else []
            usable_uniques = [uc for uc in uniques if all(col in common_columns for col in uc)]

            if not usable_pk_cols and not usable_uniques:
                # No usable constraints, just INSERT
                merge_sql = f"""
                    INSERT INTO "{schema}"."{target_table_name}" ({select_common_columns})
                    SELECT {select_common_columns} FROM "{clean_table_name}"
                """
            else:
                # Build MERGE with usable constraints only
                usable_constraints = [usable_pk_cols] if usable_pk_cols else []
                usable_constraints.extend(usable_uniques)

                constraint_conditions = []
                for constraint_cols in usable_constraints:
                    if len(constraint_cols) == 1:
                        col = constraint_cols[0]
                        constraint_conditions.append(f'tgt."{col}" = src."{col}"')
                    else:
                        multi_conditions = []
                        for col in constraint_cols:
                            multi_conditions.append(f'tgt."{col}" = src."{col}"')
                        constraint_conditions.append(f"({' AND '.join(multi_conditions)})")

                join_on_clause = " OR ".join(constraint_conditions)

                # Build UPDATE SET clause (exclude PK columns that are usable)
                update_columns = [col for col in common_columns if col not in (usable_pk_cols or [])]
                if update_columns:
                    update_set_clause = ", ".join([f'"{col}" = src."{col}"' for col in update_columns])
                else:
                    # If all common columns are PK, create a dummy update
                    update_set_clause = f'"{common_columns[0]}" = src."{common_columns[0]}"'

                values_clause = ", ".join([f'src."{col}"' for col in common_columns])

                merge_sql = f"""
                    MERGE INTO "{schema}"."{target_table_name}" AS tgt
                    USING "{clean_table_name}" AS src
                    ON ({join_on_clause})
                    WHEN MATCHED THEN
                        UPDATE SET {update_set_clause}
                    WHEN NOT MATCHED THEN
                        INSERT ({select_common_columns})
                        VALUES ({values_clause})
                """

            logger.debug(f"Generated MERGE SQL: {merge_sql}")

            with self.engine.begin() as conn:
                result = conn.execute(text(merge_sql))
                affected_rows = result.rowcount

            logger.debug(f"MERGE operation completed successfully, affected {affected_rows} rows")
            return affected_rows

        except Exception as e:
            logger.error(f"Failed to execute MERGE operation: {e}")
            raise

    def _cleanup_temp_tables(self, *temp_table_names: Optional[str]) -> None:
        """
        Clean up multiple temporary tables by dropping them from the database.

        Args:
            temp_table_names: Variable number of temporary table names to drop.
                            None or empty strings are safely ignored.

        Note:
            This method uses DROP TABLE IF EXISTS to avoid errors if tables
            don't exist. Failures are logged but don't raise exceptions to
            ensure cleanup attempts don't interrupt the main operation flow.
        """
        try:
            with self.engine.begin() as conn:
                for temp_table_name in temp_table_names:
                    if temp_table_name:  # Only drop if name is not None/empty
                        conn.execute(text(f'DROP TABLE IF EXISTS "{temp_table_name}"'))
                        logger.debug(f"Cleaned up temporary table: {temp_table_name}")

        except Exception as e:
            logger.error(f"Failed to cleanup temporary tables: {e}")
