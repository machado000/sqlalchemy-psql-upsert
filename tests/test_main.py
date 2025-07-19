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


@pytest.fixture(scope="function")
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
    empty_df = pd.DataFrame()

    pk, uniques = upserter._get_constraints(empty_df, table_name)
    assert pk == []
    assert uniques == []

    pk, uniques = upserter._get_constraints(test_df, table_name)
    assert pk == ["id"]
    assert uniques == [['id'], ['email'], ['doc_type', 'doc_number']]

    with pytest.raises(Exception):
        upserter._get_constraints(test_df, "nonexistent_table")


def test_check_conflicts(upserter, populated_table, test_df):

    table_name = "test_table"
    constraints = [['id'], ['email'], ['doc_type', 'doc_number'], ['nonexisting_constraint'], []]

    conflict_sets = []

    for keyset in constraints:
        conflict_dict = upserter._check_conflicts(test_df, keyset, table_name)
        conflict_sets.append(conflict_dict)

    assert conflict_sets[0] == {0: {(1,)}, 1: {(2,)}, 2: {(3,)}, 3: {(4,)}, 4: {(5,)}, 5: {(6,)}, 6: {(7,)}, 7: {
        (8,)}, 8: {(9,)}, 9: {(1,)}, 12: {(4,)}, 13: {(6,)}}  # return for 'id'
    assert conflict_sets[1] == {0: {(1,)}, 1: {(2,)}, 2: {(3,)}, 3: {(4,)}, 4: {(5,)}, 5: {(6,)}, 6: {(7,)}, 7: {
        (8,)}, 8: {(9,)}, 10: {(2,)}, 12: {(5,)}, 14: {(8,)}}  # return for 'email'
    assert conflict_sets[2] == {0: {(1,)}, 1: {(2,)}, 2: {(3,)}, 3: {(4,)}, 4: {(5,)}, 5: {(6,)}, 6: {(7,)}, 7: {
        (8,)}, 8: {(9,)}, 11: {(3,)}, 13: {(7,)}, 14: {(9,)}}  # return for 'doc_type', 'doc_number'
    assert conflict_sets[3] == {}  # return for 'nonexisting_constraint'
    assert conflict_sets[4] == {}  # return for 'empty_constraint'

    empty_df = pd.DataFrame()
    empty_keyset = []

    empty_df_result = upserter._check_conflicts(empty_df, constraints[0], table_name)
    assert empty_df_result == {}

    empty_keyset_result = upserter._check_conflicts(test_df, empty_keyset, table_name)
    assert empty_keyset_result == {}

    with pytest.raises(Exception):
        upserter._check_conflicts(test_df, constraints[0], "nonexisting_table")


def test_remove_multi_conflict_rows(upserter, populated_table, test_df):
    table_name = "test_table"
    expected_df = test_df[0:12].copy()

    result_df = upserter._remove_multi_conflict_rows(test_df, table_name)
    pd.testing.assert_frame_equal(result_df.sort_values(by="id").reset_index(drop=True),
                                  expected_df.sort_values(by="id").reset_index(drop=True))

    empty_df = pd.DataFrame()

    empty_df_result = upserter._remove_multi_conflict_rows(empty_df, table_name)
    pd.testing.assert_frame_equal(empty_df, empty_df_result)

    with pytest.raises(Exception):
        upserter._remove_multi_conflict_rows(test_df, "nonexistent_table")


def test_upsert_dataframe(upserter, populated_table, test_df):
    table_name = "test_table"

    result = upserter.upsert_dataframe(test_df, table_name)
    assert result is True

    empty_df = pd.DataFrame()

    result = upserter.upsert_dataframe(empty_df, table_name)
    assert result is True

    with pytest.raises(Exception):
        upserter.upsert_dataframe(test_df, "nonexistent_table")
