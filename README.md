# SQLAlchemy PostgreSQL Upsert

[![PyPI version](https://img.shields.io/pypi/v/sqlalchemy-psql-upsert)](https://pypi.org/project/sqlalchemy-psql-upsert/)
[![License](https://img.shields.io/github/license/machado000/sqlalchemy-psql-upsert)](https://github.com/machado000/sqlalchemy-psql-upsert/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/machado000/sqlalchemy-psql-upsert)](https://github.com/machado000/sqlalchemy-psql-upsert/issues)
[![Last Commit](https://img.shields.io/github/last-commit/machado000/sqlalchemy-psql-upsert)](https://github.com/machado000/sqlalchemy-psql-upsert/commits/main)


A high-performance Python library for PostgreSQL UPSERT operations with intelligent conflict resolution using PostgreSQL temporary tables and atomic MERGE statements. Designed for reliability, data integrity, and modern Python development.


## 🚀 Features

- **Temporary Table Staging**: Uses PostgreSQL temporary tables for efficient, isolated staging and conflict analysis
- **Atomic MERGE Operations**: Single-transaction upserts using PostgreSQL 15+ MERGE statements for reliability and performance
- **Multi-constraint Support**: Handles primary keys, unique constraints, and composite constraints simultaneously
- **Intelligent Conflict Resolution**: Automatically filters ambiguous conflicts and deduplicates data
- **Automatic NaN to NULL Conversion**: Seamlessly converts pandas NaN/None values to PostgreSQL NULL values
- **Schema Validation**: Automatic table and column validation before operations
- **Comprehensive Logging**: Detailed debug information and progress tracking
- **Modern Typing**: Fully typed with Python 3.10+ type hints


## 📦 Installation

### Using Poetry (Recommended)
```bash
poetry add sqlalchemy-psql-upsert
```

### Using pip
```bash
pip install sqlalchemy-psql-upsert
```

## ⚙️ Configuration

### Database Privileges Requirements

Besides SELECT, INSERT, UPDATE permissions on target tables, this library requires PostgreSQL `TEMPORARY` privilege to function properly:

**Why Temporary Tables?**
- **Isolation**: Staging data doesn't interfere with production tables during analysis
- **Performance**: Bulk operations are faster on temporary tables
- **Safety**: Failed operations don't leave partial data in target tables
- **Atomicity**: Entire upsert operation happens in a single transaction

### Environment Variables

Create a `.env` file or set the following environment variables:

```bash
# PostgreSQL Configuration
PGHOST = localhost
PGPORT = 5432
PGDATABASE = your_database
PGUSER = your_username
PGPASSWORD = your_password
```

### Configuration Class

```python
from sqlalchemy_psql_upsert import PgConfig

# Default configuration from environment
config = PgConfig()

# Manual configuration
config = PgConfig(
    host="localhost",
    port="5432",
    user="myuser",
    password="mypass",
    dbname="mydb"
)

print(config.uri())  # postgresql+psycopg2://myuser:mypass@localhost:5432/mydb
```

## 🛠️ Quick Start

### Connection Testing

```python
from sqlalchemy_psql_upsert import test_connection

# Test default connection
success, message = test_connection()
if success:
    print("✅ Database connection OK")
else:
    print(f"❌ Connection failed: {message}")
```

### Privileges Verification

**Important**: This library requires `CREATE TEMP TABLE` privileges to function properly. The client automatically verifies these privileges during initialization.

```python
from sqlalchemy_psql_upsert import PostgresqlUpsert, PgConfig
from sqlalchemy import create_engine

# Test connection and privileges
config = PgConfig()

try:
    # This will automatically test temp table privileges
    upserter = PostgresqlUpsert(config=config)
    print("✅ Connection and privileges verified successfully")
    
except PermissionError as e:
    print(f"❌ Privilege error: {e}")
    print("Solution: Grant temporary privileges with:")
    print("GRANT TEMPORARY ON DATABASE your_database TO your_user;")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

**Grant Required Privileges:**
```sql
-- As database administrator, grant temporary table privileges
GRANT TEMPORARY ON DATABASE your_database TO your_user;

-- Alternatively, grant more comprehensive privileges
GRANT CREATE ON DATABASE your_database TO your_user;
```

### Basic Usage

```python
import pandas as pd
from sqlalchemy_psql_upsert import PostgresqlUpsert, PgConfig

# Configure database connection
config = PgConfig()  # Loads from environment variables
upserter = PostgresqlUpsert(config=config)

# Prepare your data
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com']
})

# Perform upsert
success, affected_rows = upserter.upsert_dataframe(
    dataframe=df,
    table_name='users',
    schema='public'
)

print(f"✅ Upserted {affected_rows} rows successfully")
```

### Advanced Configuration

```python
from sqlalchemy import create_engine

# Using custom SQLAlchemy engine
engine = create_engine('postgresql://user:pass@localhost:5432/mydb')
upserter = PostgresqlUpsert(engine=engine, debug=True)

# Upsert with custom schema
success, affected_rows = upserter.upsert_dataframe(
    dataframe=large_df,
    table_name='products',
    schema='inventory'
)
```

### Data Type Handling

The library automatically handles pandas data type conversions:

```python
import pandas as pd
import numpy as np

# DataFrame with NaN values
df = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', None, 'David'],      # None values
    'score': [85.5, np.nan, 92.0, 88.1],         # NaN values
    'active': [True, False, None, True]           # Mixed types with None
})

# All NaN and None values are automatically converted to PostgreSQL NULL
upserter.upsert_dataframe(df, 'users')
# Result: NaN/None → NULL in PostgreSQL
```

## 🔍 How It Works


### Detailed Upsert Workflow

This library uses a robust multi-step approach for reliable, high-performance upserts:

1. **Temporary Table Staging**
   - A temporary table is created with the same structure as the target table, plus a `skip_reason` column.
   - All input DataFrame rows are bulk inserted into this table.

2. **Multi-Row Conflict Detection**
   - Each row in the temp table is checked for conflicts against the target table's constraints (PK, unique, composite unique).
   - Rows that would conflict with more than one target row are marked with `skip_reason = 'multiple_target_conflicts'`.

3. **Ranking and Deduplication**
   - For rows that only conflict with a single target row, the library checks for duplicate source rows (multiple input rows targeting the same record).
   - Only the last row for each target is kept; others are marked with `skip_reason = 'duplicate_source_rows'`.

4. **Constructing the Clean Rows Table**
   - A second temporary table (the "clean table") is created with the same structure as the target table.
   - All rows from the raw temp table where `skip_reason IS NULL` are copied into the clean table.

5. **Merging Clean Table into Target Table**
   - A single, atomic PostgreSQL `MERGE` statement is executed:
     - If a row in the clean table matches a row in the target table (by any constraint), it is updated.
     - If there is no match, the row is inserted.

6. **Fetching and Returning Conflicts**
   - After the merge, all rows from the raw temp table where `skip_reason IS NOT NULL` are fetched and returned to the user, along with the reason they were skipped.

This process ensures only valid, deduplicated rows are upserted, and all conflicts are tracked and reported for further review.


### Constraint Detection

The library automatically analyzes your target table to identify:
- **Primary key constraints**: Single or composite primary keys
- **Unique constraints**: Single column unique constraints
- **Composite unique constraints**: Multi-column unique constraints

All constraint types are handled simultaneously in a single operation.


## 🚨 Pros, Cons & Considerations

### ✅ Advantages of Temporary Table + CTE + MERGE Approach

**Performance Benefits:**
- **Single Transaction**: Entire operation is atomic, no partial updates or race conditions
- **Bulk Operations**: High-performance bulk inserts into temporary tables
- **Efficient Joins**: PostgreSQL optimizes joins between temporary and main tables
- **Minimal Locking**: Temporary tables don't interfere with concurrent operations

**Reliability Benefits:**
- **Comprehensive Conflict Resolution**: Handles all constraint types simultaneously
- **Deterministic Results**: Same input always produces same output
- **Automatic Cleanup**: Temporary tables are automatically dropped
- **ACID Compliance**: Full transaction safety and rollback capability

**Data Integrity Benefits:**
- **Ambiguity Detection**: Automatically detects and skips problematic rows
- **Deduplication**: Handles duplicate input data intelligently
- **Constraint Validation**: PostgreSQL validates all constraints during MERGE

### ❌ Limitations and Trade-offs

**Resource Requirements:**
- **Memory Usage**: All input data is staged in temporary tables (memory-resident)
- **Temporary Space**: Requires sufficient temporary storage for staging tables
- **Single-threaded**: No parallel processing (traded for reliability and simplicity)

**PostgreSQL Specifics:**
- **Version Dependency**: MERGE statement requires PostgreSQL 15+
- **Session-based Temp Tables**: Temporary tables are tied to database sessions

**Privilege-related Limitations:**
- **Database Administrator**: May need DBA assistance to grant `TEMPORARY` privileges
- **Shared Hosting**: Some cloud providers restrict temporary table creation
- **Security Policies**: Corporate environments may restrict temporary table usage

**Scale Considerations:**
- **Large Dataset Handling**: Very large datasets (>1M rows) may require memory tuning
- **Transaction Duration**: Entire operation happens in one transaction (longer lock times)

### 🎯 Best Practices

**Memory Management:**
```python
# For large datasets, monitor memory usage
import pandas as pd

# Consider chunking very large datasets manually if needed
def chunk_dataframe(df, chunk_size=50000):
    for start in range(0, len(df), chunk_size):
        yield df[start:start + chunk_size]

# Process in manageable chunks
for chunk in chunk_dataframe(large_df):
    success, rows = upserter.upsert_dataframe(chunk, 'target_table')
    print(f"Processed chunk: {rows} rows")
```

**Performance Optimization:**
```python
# Ensure proper indexing on conflict columns
# CREATE INDEX idx_email ON target_table(email);
# CREATE INDEX idx_composite ON target_table(doc_type, doc_number);

# Use debug mode to monitor performance
upserter = PostgresqlUpsert(engine=engine, debug=True)
```

**Error Handling:**
```python
try:
    success, affected_rows = upserter.upsert_dataframe(df, 'users')
    logger.info(f"Successfully upserted {affected_rows} rows")
except ValueError as e:
    logger.error(f"Validation error: {e}")
except Exception as e:
    logger.error(f"Upsert failed: {e}")
    # Handle rollback - transaction is automatically rolled back
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `pytest tests/ -v`
5. **Submit a pull request**


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/machado000/sqlalchemy-psql-upsert/issues)
- **Documentation**: Check the docstrings and test files for detailed usage examples