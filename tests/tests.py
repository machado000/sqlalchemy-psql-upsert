import pytest
import pandas as pd
import pandas.testing as pdt
from sqlalchemy import create_engine, text, Table, Column, Integer, String, MetaData, UniqueConstraint
from sqlalchemy_upsert.main import PostgresqlUpsert
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="module")
def upserter(pg_engine):
    return PostgresqlUpsert(pg_engine)


@pytest.fixture(scope="module")
def mock_table(pg_engine):
    metadata = MetaData()
    table = Table(
        "mock_upsert", metadata,
        Column("id", Integer, primary_key=True),
        Column("email", String, unique=True),
        Column("doc_type", String),
        Column("doc_number", String),
        UniqueConstraint("doc_type", "doc_number"),
    )
    metadata.create_all(pg_engine)
    return table


@pytest.fixture
def upserter(pg_engine):
    return PostgresqlUpsert(engine=pg_engine)


def test_empty_dataframe_skips(upserter, caplog):
    df = pd.DataFrame()
    result = upserter.upsert_dataframe(df, "test_table")
    assert result is True
    assert "empty DataFrame" in caplog.text


def test_missing_table_fails(upserter):
    df = pd.DataFrame({"id": [1], "value": ["a"]})
    result = upserter.upsert_dataframe(df, "nonexistent_table")
    assert result is False


def test_upsert_dataframe_deduplication(upserter, mock_table):
    df = pd.DataFrame([
        {"id": 1, "email": "a@x.com", "doc_type": "CPF", "doc_number": "111"},
        {"id": 2, "email": "a@x.com", "doc_type": "RG",  "doc_number": "222"},  # conflict on email
        {"id": 3, "email": "c@x.com", "doc_type": "CPF", "doc_number": "111"},  # conflict on doc_type+doc_number
        {"id": 1, "email": "d@x.com", "doc_type": "CNH", "doc_number": "333"},  # conflict on id
        {"id": 4, "email": "e@x.com", "doc_type": "RG",  "doc_number": "555"},  # no conflict
        {"id": 5, "email": "f@x.com", "doc_type": "RG",  "doc_number": "222"},  # no conflict
    ])

    result = upserter.upsert_dataframe(df, "mock_upsert", deduplicate=True, chunk_size=2)

    assert result is True

    # Fetch from DB to verify
    with upserter.engine.connect() as conn:
        rows = conn.execute(mock_table.select()).fetchall()

    # Only 3 rows should survive after deduplication
    assert len(rows) == 3
    emails = [r["email"] for r in rows]
    assert "a@x.com" in emails
    assert "e@x.com" in emails
    assert "f@x.com" in emails


def test_get_constraints_order(upserter):
    df = pd.DataFrame({"id": [1], "email": ["x"], "doc_type": ["CPF"], "doc_number": ["111"]})
    table = Table("test_table", MetaData(), autoload_with=upserter.engine)
    pk, constraints = upserter._get_constraints(df, table)
    assert pk == ["id"]
    assert constraints[0] == ["id"]


def test_insert_fallback_on_no_constraints(upserter, caplog):
    upserter.engine.execute("DROP TABLE IF EXISTS test_table")
    upserter.engine.execute("""
        CREATE TABLE test_table (
            id INTEGER,
            email TEXT,
            doc_type TEXT,
            doc_number TEXT
        )
    """)
    df = pd.DataFrame({"email": ["a"], "doc_type": ["X"], "doc_number": ["000"]})
    upserter.upsert_dataframe(df, "test_table")
    assert "Performing plain INSERT" in caplog.text


def test_upsert_with_constraints(upserter):
    upserter.engine.execute("DROP TABLE IF EXISTS test_table")
    upserter.engine.execute("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE,
            doc_type TEXT,
            doc_number TEXT,
            UNIQUE(doc_type, doc_number)
        )
    """)
    df = pd.DataFrame([
        {"id": 1, "email": "foo@x.com", "doc_type": "CPF", "doc_number": "111"},
        {"id": 2, "email": "bar@x.com", "doc_type": "CPF", "doc_number": "111"},
        {"id": 3, "email": "bar@x.com", "doc_type": "RG",  "doc_number": "222"},
    ])
    upserter.upsert_dataframe(df, "test_table")
    count = upserter.engine.execute("SELECT COUNT(*) FROM test_table").scalar()
    assert count == 2
