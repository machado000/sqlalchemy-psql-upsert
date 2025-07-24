import pytest
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import Table, Column, Integer, String, MetaData, UniqueConstraint
from sqlalchemy_psql_upsert.client import PostgresqlUpsert
from testcontainers.postgres import PostgresContainer
import logging

# Configure logging to show debug messages during tests
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    force=True  # Override any existing logging configuration
)

# Ensure the client logger shows debug messages
logger = logging.getLogger('sqlalchemy_psql_upsert.client')
logger.setLevel(logging.INFO)
logger.propagate = True


@pytest.fixture(scope="module")
def pg_engine():
    with PostgresContainer("postgres:15") as pg:
        engine = create_engine(pg.get_connection_url())
        metadata = MetaData()
        Table(
            "target_table", metadata,
            Column("id", Integer, nullable=False, primary_key=True),
            Column("email", String, nullable=True, unique=True),
            Column("doc_type", String, nullable=True),
            Column("doc_number", String, nullable=True),
            Column("count_number", Integer, nullable=True),
            UniqueConstraint("doc_type", "doc_number", name="uq_doc"),
        )
        metadata.create_all(engine)
        yield engine


@pytest.fixture(scope="module")
def upserter(pg_engine):
    # Initialize with debug=True to enable debug logging
    return PostgresqlUpsert(engine=pg_engine, debug=False)


@pytest.fixture(scope="function")
def populated_table(pg_engine):
    initial_data = pd.DataFrame([
        {"id": 1, "email": "a@x.com", "doc_type": "CPF", "doc_number": "901", "count_number": 0},
        {"id": 2, "email": "b@x.com", "doc_type": "RG",  "doc_number": "902", "count_number": 0},
        {"id": 3, "email": "c@x.com", "doc_type": "CNH", "doc_number": "903", "count_number": 0},
        {"id": 4, "email": "d@x.com", "doc_type": "CPF", "doc_number": "904", "count_number": 0},
        {"id": 5, "email": "e@x.com", "doc_type": "RG", "doc_number": "905", "count_number": 0},
        {"id": 6, "email": "f@x.com", "doc_type": "CNH", "doc_number": "906", "count_number": 0},
        {"id": 7, "email": "g@x.com", "doc_type": "CPF", "doc_number": "907", "count_number": 0},
        {"id": 8, "email": "h@x.com", "doc_type": "RG", "doc_number": "908", "count_number": 0},
        {"id": 9, "email": "i@x.com", "doc_type": "CNH", "doc_number": "909", "count_number": 0},
    ])
    with pg_engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE target_table"))
        initial_data.to_sql("target_table", conn, if_exists="append", index=False)
    yield


@pytest.fixture(scope="function")
def test_df():
    test_df = pd.DataFrame([
        {"id": 1, "email": "a@x.com", "doc_type": "CPF", "doc_number": "911", "count_number": 108},
        {"id": 2, "email": "b@x.com", "doc_type": "RG",  "doc_number": "912", "count_number": 140},
        {"id": 3, "email": "c@x.com", "doc_type": "CNH", "doc_number": "913", "count_number": 156},
        {"id": 4, "email": "d@x.com", "doc_type": "CPF", "doc_number": "914", "count_number": 139},
        {"id": 5, "email": "e@x.com", "doc_type": "RG", "doc_number": "915", "count_number": 121},
        {"id": 6, "email": "f@x.com", "doc_type": "CNH", "doc_number": "916", "count_number": 160},
        {"id": 7, "email": "g@x.com", "doc_type": "CPF", "doc_number": "917", "count_number": 171},
        {"id": 8, "email": "h@x.com", "doc_type": "RG", "doc_number": "918", "count_number": 131},
        {"id": 9, "email": "i@x.com", "doc_type": "CNH", "doc_number": "919", "count_number": 196},
        {"id": 1, "email": "j@x.com", "doc_type": "CPF", "doc_number": "920", "count_number": 184},
        {"id": 11, "email": "b@x.com", "doc_type": "RG", "doc_number": "921", "count_number": 114},
        {"id": 12, "email": "l@x.com", "doc_type": "CNH", "doc_number": "903", "count_number": 111},
        {"id": 4, "email": "e@x.com", "doc_type": "CPF", "doc_number": "913", "count_number": 171},
        {"id": 6, "email": "n@x.com", "doc_type": "CPF", "doc_number": "907", "count_number": 121},
        {"id": 15, "email": "h@x.com", "doc_type": "CNH", "doc_number": "909", "count_number": 119},
        {"id": 7, "email": "g@x.com", "doc_type": "CPF", "doc_number": "927", "count_number": 116},
        {"id": 7, "email": "g@x.com", "doc_type": "CPF", "doc_number": "937", "count_number": 117},
        {"id": 18, "email": "r@x.com", "doc_type": "CNH", "doc_number": "938", "count_number": 162},
        {"id": 19, "email": "s@x.com", "doc_type": "RG", "doc_number": "939", "count_number": 181},
        {"id": 20, "email": "t@x.com", "doc_type": "CPF", "doc_number": "940", "count_number": 199},
    ])
    yield test_df


def test_get_constraints(upserter, populated_table):
    table_name = "target_table"

    pk, uniques = upserter._get_constraints(table_name)
    assert pk == ["id"]
    assert uniques == [['doc_type', 'doc_number'], ['email']]

    with pytest.raises(Exception):
        upserter._get_constraints("nonexistent_table")


def test_get_dataframe_constraints(upserter, test_df):
    table_name = "target_table"
    empty_df = pd.DataFrame()

    pk, uniques = upserter._get_dataframe_constraints(empty_df, table_name)
    assert pk == []
    assert uniques == []

    pk, uniques = upserter._get_dataframe_constraints(test_df, table_name)
    assert pk == ["id"]
    assert uniques == [['doc_type', 'doc_number'], ['email']]

    with pytest.raises(Exception):
        upserter._get_dataframe_constraints(test_df, "nonexistent_table")


def test_upsert_dataframe(upserter, populated_table, test_df):
    table_name = "target_table"

    result, affected_rows = upserter.upsert_dataframe(test_df, table_name)

    logger.info(f"Upsert result: {result}, Affected rows: {affected_rows}")

    with upserter.engine.connect() as conn:
        result_df = pd.read_sql("SELECT * FROM target_table ORDER BY id", conn)
        print(result_df)

    assert result is True

    # empty_df = pd.DataFrame()

    # result = upserter.upsert_dataframe(empty_df, table_name)
    # assert result is True

    # with pytest.raises(Exception):
    #     upserter.upsert_dataframe(test_df, "nonexistent_table")
