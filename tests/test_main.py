import pytest
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import Table, Column, Integer, String, MetaData, UniqueConstraint
from sqlalchemy_upsert.main import PostgresqlUpsert
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="module")
def pg_engine():
    with PostgresContainer("postgres:15") as pg:
        engine = create_engine(pg.get_connection_url())
        metadata = MetaData()
        Table(
            "test_table", metadata,
            Column("id", Integer, primary_key=True),
            Column("email", String, unique=True),
            Column("doc_type", String),
            Column("doc_number", String),
            UniqueConstraint("doc_type", "doc_number", name="uq_doc"),
        )
        metadata.create_all(engine)
        yield engine


@pytest.fixture(scope="module")
def upserter(pg_engine):
    return PostgresqlUpsert(engine=pg_engine)


@pytest.fixture(scope="function")
def populated_table(pg_engine):
    initial_data = pd.DataFrame([
        {"id": 1, "email": "a@x.com", "doc_type": "CPF", "doc_number": "901"},
        {"id": 2, "email": "b@x.com", "doc_type": "RG",  "doc_number": "902"},
        {"id": 3, "email": "c@x.com", "doc_type": "CNH", "doc_number": "903"},
        {"id": 4, "email": "d@x.com", "doc_type": "CPF", "doc_number": "904"},
        {"id": 5, "email": "e@x.com", "doc_type": "RG", "doc_number": "905"},
        {"id": 6, "email": "f@x.com", "doc_type": "CNH", "doc_number": "906"},
        {"id": 7, "email": "g@x.com", "doc_type": "CPF", "doc_number": "907"},
        {"id": 8, "email": "h@x.com", "doc_type": "RG", "doc_number": "908"},
        {"id": 9, "email": "i@x.com", "doc_type": "CNH", "doc_number": "909"},
    ])
    with pg_engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE test_table"))
        initial_data.to_sql("test_table", conn, if_exists="append", index=False)
    yield


@pytest.fixture(scope="module")
def test_df():
    test_df = pd.DataFrame([
        {"id": 1, "email": "a@x.com", "doc_type": "CPF", "doc_number": "901"},
        {"id": 2, "email": "b@x.com", "doc_type": "RG",  "doc_number": "902"},
        {"id": 3, "email": "c@x.com", "doc_type": "CNH", "doc_number": "903"},
        {"id": 4, "email": "d@x.com", "doc_type": "CPF", "doc_number": "904"},
        {"id": 5, "email": "e@x.com", "doc_type": "RG", "doc_number": "905"},
        {"id": 6, "email": "f@x.com", "doc_type": "CNH", "doc_number": "906"},
        {"id": 7, "email": "g@x.com", "doc_type": "CPF", "doc_number": "907"},
        {"id": 8, "email": "h@x.com", "doc_type": "RG", "doc_number": "908"},
        {"id": 9, "email": "i@x.com", "doc_type": "CNH", "doc_number": "909"},
        {"id": 1, "email": "j@x.com", "doc_type": "CPF", "doc_number": "910"},  # conflict on id
        {"id": 11, "email": "b@x.com", "doc_type": "RG", "doc_number": "911"},  # conflict on email
        {"id": 12, "email": "l@x.com", "doc_type": "CNH", "doc_number": "903"},  # conflict on doc_type_number
        {"id": 4, "email": "e@x.com", "doc_type": "CPF", "doc_number": "913"},  # conflict on id + email
        {"id": 6, "email": "n@x.com", "doc_type": "CPF", "doc_number": "907"},  # conflict on id + doc_type_number
        {"id": 15, "email": "h@x.com", "doc_type": "CNH", "doc_number": "909"},  # conflict on email + doc_type_number
    ])
    yield test_df


def test_get_constraints(upserter, test_df):
    table_name = "test_table"
    pk, uniques = upserter._get_constraints(test_df, table_name)
    assert pk == ["id"]
    assert uniques == [['id'], ['email'], ['doc_type', 'doc_number']]

    empty_df = pd.DataFrame()
    pk, uniques = upserter._get_constraints(empty_df, table_name)
    assert pk == []
    assert uniques == []


def test_check_conflicts(upserter, populated_table, test_df):
    table_name = "test_table"
    constraints = [['id'], ['email'], ['doc_type', 'doc_number'], ['nonexisting']]

    conflict_sets = []

    for keyset in constraints:
        conflict_index = upserter._check_conflicts(test_df, keyset, table_name)
        print(f"Conflicts for {keyset}: {conflict_index.tolist()}")
        conflict_sets.append(conflict_index)

    assert conflict_sets[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13]  # return for 'id'
    assert conflict_sets[1].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14]  # return for 'email'
    assert conflict_sets[2].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14]  # return for 'doc_type', 'doc_number'
    assert conflict_sets[3].tolist() == []  # return for 'nonexisting'


def test_remove_multi_conflict_rows(upserter, populated_table, test_df):
    table_name = "test_table"

    result_df = upserter._remove_multi_conflict_rows(test_df, table_name)
    expected_df = test_df[0:11].copy()

    pd.testing.assert_frame_equal(result_df.sort_values(by="id").reset_index(drop=True),
                                  expected_df.sort_values(by="id").reset_index(drop=True))


def test_empty_dataframe_skips(upserter, caplog):
    df = pd.DataFrame()
    result = upserter.upsert_dataframe(df, "test_table")
    assert result is True
    assert "empty DataFrame" in caplog.text


def test_missing_table_fails(upserter):
    df = pd.DataFrame({"id": [1], "value": ["a"]})
    result = upserter.upsert_dataframe(df, "nonexistent_table")
    assert result is False


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
