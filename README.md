# SQLAlchemy PostgreSQL Upsert

[![PyPI version](https://img.shields.io/pypi/v/sqlalchemy-psql-upsert)](https://pypi.org/project/sqlalchemy-psql-upsert/)
[![License](https://img.shields.io/github/license/machado000/sqlalchemy-psql-upsert)](https://github.com/machado000/sqlalchemy-psql-upsert/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/machado000/sqlalchemy-psql-upsert)](https://github.com/machado000/sqlalchemy-psql-upsert/issues)
[![Last Commit](https://img.shields.io/github/last-commit/machado000/sqlalchemy-psql-upsert)](https://github.com/machado000/sqlalchemy-psql-upsert/commits/main)


A Python library for intelligent PostgreSQL upsert operations with advanced conflict resolution and multi-threaded processing.

## 🆕 What's New in v1.0.3

- **Automatic NaN to NULL conversion**: All pandas NaN values (np.nan, pd.NaType, None) are now automatically converted to PostgreSQL NULL values during upsert operations
- **Improved data integrity**: Better handling of missing/null data in DataFrames

## 🚀 Features

- **Multi-constraint conflict detection**: Automatically handles primary key, unique constraints, and composite constraints
- **Smart conflict filtering**: Removes rows that would conflict with multiple existing records
- **Automatic NaN to NULL conversion**: Seamlessly converts pandas NaN values to PostgreSQL NULL values
- **Multi-threaded processing**: Parallel chunk processing for large datasets
- **Configurable batch sizes**: Optimize memory usage and processing speed
- **Schema validation**: Automatic table and column validation before operations
- **Comprehensive error handling**: Detailed logging and error reporting

## 📦 Installation

### Using Poetry (Recommended)
```bash
poetry add sqlalchemy-psql-upsert
```

### Using pip
```bash
pip install sqlalchemy-psql-upsert
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
success = upserter.upsert_dataframe(
    dataframe=df,
    table_name='users',
    schema='public',
    chunk_size=10000,
    max_workers=4
)
```

### Advanced Configuration

```python
from sqlalchemy import create_engine

# Using custom SQLAlchemy engine
engine = create_engine('postgresql://user:pass@localhost:5432/mydb')
upserter = PostgresqlUpsert(engine=engine, debug=True)

# Custom upsert with options
upserter.upsert_dataframe(
    dataframe=large_df,
    table_name='products',
    schema='inventory',
    chunk_size=5000,           # Smaller chunks for memory efficiency
    max_workers=8,             # More workers for better parallelism
    remove_multi_conflict_rows=True  # Remove problematic rows
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

## ⚙️ Configuration

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

## 🔍 How It Works

### Constraint Detection
The library automatically analyzes your target table to identify:
- Primary key constraints
- Unique constraints  
- Composite unique constraints

### Conflict Resolution Process
1. **Constraint Analysis**: Identifies all relevant constraints on the target table
2. **Conflict Detection**: For each constraint, finds DataFrame rows that would conflict with existing data
3. **Multi-Conflict Filtering**: Removes rows that would match multiple existing records (ambiguous conflicts)
4. **Intelligent Upsert**: Uses PostgreSQL's `ON CONFLICT` clause with appropriate constraint targeting

### Example Conflict Scenarios

Consider a table with these constraints:
- Primary key: `id`
- Unique constraint: `email`
- Composite unique constraint: `(doc_type, doc_number)`

```python
# This row conflicts on 'id' only - will be upserted
{'id': 1, 'email': 'new@example.com', 'doc_type': 'CPF', 'doc_number': '123'}

# This row conflicts on both 'id' and 'email' - will be removed
{'id': 1, 'email': 'existing@example.com', 'doc_type': 'RG', 'doc_number': '456'}
```

## 🚨 Limitations & Considerations

### Current Limitations
- **PostgreSQL only**: Currently supports PostgreSQL databases exclusively
- **Memory usage**: Large datasets are processed in memory (chunked processing helps)
- **Complex constraints**: Some exotic PostgreSQL constraint types may not be fully supported
- **Transaction scope**: Each chunk is processed in its own transaction

### Best Practices
- **Data preparation**: NaN values are automatically converted to NULL - no manual preprocessing needed
- **Chunk sizing**: Start with 10,000 rows per chunk, adjust based on your data and hardware
- **Worker count**: Use 2-4 workers per CPU core, but test with your specific workload
- **Memory monitoring**: Monitor memory usage with large datasets
- **Index considerations**: Ensure proper indexing on conflict columns for optimal performance

## � Testing

Run the comprehensive test suite:

```bash
# Install with development dependencies
pip install sqlalchemy-psql-upsert[dev]

# Run tests
pytest tests/ -v
```

The test suite uses TestContainers to spin up real PostgreSQL instances, ensuring tests run against actual database constraints.

## 🔧 API Reference

### Core Classes

#### `PostgresqlUpsert`
Main upsert client with intelligent conflict resolution.

```python
class PostgresqlUpsert:
    def __init__(self, config=None, engine=None, debug=False):
        """Initialize PostgreSQL upsert client."""
    
    def upsert_dataframe(self, dataframe, table_name, schema="public", 
                        chunk_size=10000, max_workers=4, 
                        remove_multi_conflict_rows=True) -> bool:
        """Upsert DataFrame with conflict resolution."""
    
    def list_tables(self) -> List[str]:
        """Get list of all tables in database."""
```

#### `PgConfig`
PostgreSQL connection configuration.

```python
@dataclass
class PgConfig:
    host: str = os.getenv('PGHOST', '')
    port: str = os.getenv('PGPORT', '5432') 
    dbname: str = os.getenv('PGDATABASE', '')
    user: str = os.getenv('PGUSER', '')
    password: str = os.getenv('PGPASSWORD', '')
    
    def uri(self) -> str:
        """Generate connection URI string."""
```

### Utility Functions

#### `quick_upsert(dataframe, table_name, connection_string=None, **kwargs) -> bool`
One-line upsert for simple use cases.

#### `test_connection(config=None, engine=None) -> Tuple[bool, str]`
Test database connectivity and return (success, message).

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
- **Discussions**: [GitHub Discussions](https://github.com/machado000/sqlalchemy-psql-upsert/discussions)
- **Documentation**: Check the docstrings and test files for detailed usage examples