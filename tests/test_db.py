import sqlite3

import pytest

from frugal_analyst import db


@pytest.fixture(scope="module")
def db_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("dbs")
    db.ensure_dbs(path)
    return path


def _query(db_dir, domain, sql):
    conn = sqlite3.connect(db.db_path(domain, db_dir))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _tables(db_dir, domain):
    return [r[0] for r in _query(db_dir, domain, "SELECT name FROM sqlite_master WHERE type = 'table'")]


def test_domains():
    assert db.DOMAINS == ["retail", "clinic", "logistics", "school", "saas"]
    assert db.TRAIN_DOMAINS == ["retail", "clinic", "logistics", "school"]
    assert db.OOD_DOMAINS == ["saas"]


@pytest.mark.parametrize("domain", db.DOMAINS)
def test_table_sizes(db_dir, domain):
    tables = _tables(db_dir, domain)
    assert 3 <= len(tables) <= 4 + 1  # 3 or 4 data tables, plus one small lookup table
    sizes = {t: _query(db_dir, domain, f"SELECT COUNT(*) FROM {t}")[0][0] for t in tables}
    big = [t for t, n in sizes.items() if n >= 300]
    assert len(big) >= 3, sizes
    assert all(n <= 3000 for n in sizes.values()), sizes


@pytest.mark.parametrize("domain", db.DOMAINS)
def test_foreign_keys_hold(db_dir, domain):
    assert _query(db_dir, domain, "PRAGMA foreign_key_check") == []


@pytest.mark.parametrize("domain", db.DOMAINS)
def test_coded_columns(db_dir, domain):
    coded = db.CODED_COLUMNS[domain]
    assert len(coded) >= 2
    for table, column in coded:
        values = _query(db_dir, domain, f"SELECT DISTINCT {column} FROM {table}")
        assert 2 <= len(values) <= 8, (table, column)


@pytest.mark.parametrize("domain", db.DOMAINS)
def test_dates_are_iso_text(db_dir, domain):
    for table in _tables(db_dir, domain):
        cols = [r[1] for r in _query(db_dir, domain, f"PRAGMA table_info({table})") if r[1].endswith("_date")]
        for col in cols:
            first = _query(db_dir, domain, f"SELECT {col} FROM {table} LIMIT 1")[0][0]
            assert len(first) == 10 and first[4] == "-" and first[7] == "-"


def test_same_seed_same_data(tmp_path):
    a, b = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    for domain in db.DOMAINS:
        db.build_db(domain, a, seed=3)
        db.build_db(domain, b, seed=3)
        dump = lambda p: list(sqlite3.connect(p).iterdump())
        assert dump(a) == dump(b)


def test_different_seed_different_data(tmp_path):
    a, b = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    db.build_db("retail", a, seed=1)
    db.build_db("retail", b, seed=2)
    assert list(sqlite3.connect(a).iterdump()) != list(sqlite3.connect(b).iterdump())


def test_unknown_domain_raises(tmp_path):
    with pytest.raises(ValueError):
        db.build_db("nope", tmp_path / "x.sqlite")


def test_db_path_default_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("FRUGAL_DB_DIR", raising=False)
    assert db.db_path("retail") == db.DEFAULT_DB_DIR / "retail.sqlite"
    assert db.DEFAULT_DB_DIR.parts[-2:] == ("data", "dbs")
    monkeypatch.setenv("FRUGAL_DB_DIR", str(tmp_path))
    assert db.db_path("clinic") == tmp_path / "clinic.sqlite"
    explicit = tmp_path / "other"
    assert db.db_path("clinic", explicit) == explicit / "clinic.sqlite"  # argument wins over the env var


def test_ensure_dbs_builds_only_missing(tmp_path):
    db.ensure_dbs(tmp_path)
    marker = tmp_path / "retail.sqlite"
    before = marker.stat().st_mtime_ns
    db.ensure_dbs(tmp_path)
    assert marker.stat().st_mtime_ns == before
    marker.unlink()
    db.ensure_dbs(tmp_path)
    assert marker.exists()


@pytest.mark.parametrize("domain", db.DOMAINS)
def test_schema_text_format(db_dir, domain):
    lines = db.schema_text(domain, db_dir).splitlines()
    assert len(lines) == len(_tables(db_dir, domain))
    for line in lines:
        assert "(" in line and ")" in line and line.endswith(" rows")
    assert "CXL_USR" not in db.schema_text("retail", db_dir)  # no sample values


def test_schema_text_row_count(db_dir):
    expected = _query(db_dir, "retail", "SELECT COUNT(*) FROM orders")[0][0]
    assert f"-- {expected} rows" in db.schema_text("retail", db_dir)
