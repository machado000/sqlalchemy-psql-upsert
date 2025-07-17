# sqlalchemy-upsert

sqlalchemy-upsert is a Python package designed for efficiently upserting data from CSV files into PostgreSQL databases and dumping tables from Microsoft SQL Server (MSSQL) into CSV format. This package simplifies the process of data migration and synchronization between different database systems.

## Features

- Upsert data from CSV files into PostgreSQL.
- Dump tables from MSSQL to CSV files.
- Support for handling various data types and ensuring data integrity during upserts.
- Multi-threaded processing for improved performance.

## Installation

To install the package, you can use Poetry. First, ensure you have Poetry installed. Then, run the following command:

```bash
poetry install
```

This will install all the necessary dependencies specified in the `pyproject.toml` file.

## Usage

To use the package, you can run the main script located in `src/sqlalchemy_upsert/main.py`. Ensure that your environment variables for database connections are set correctly.

```bash
python src/sqlalchemy_upsert/main.py
```

### Environment Variables

The following environment variables need to be set for the database connections:

- `PGSQL_HOST`: Hostname for the PostgreSQL database.
- `PGSQL_PORT`: Port for the PostgreSQL database (default is 5432).
- `PGSQL_USER`: Username for the PostgreSQL database.
- `PGSQL_PASS`: Password for the PostgreSQL database.
- `PGSQL_NAME`: Name of the PostgreSQL database.
- `MSSQL_HOST`: Hostname for the MSSQL database.
- `MSSQL_PORT`: Port for the MSSQL database (default is 1433).
- `MSSQL_USER`: Username for the MSSQL database.
- `MSSQL_PASS`: Password for the MSSQL database.
- `MSSQL_DATABASE`: Name of the MSSQL database.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.