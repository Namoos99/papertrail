import pytest

from papertrail import config as C


@pytest.fixture(scope="session")
def store():
    if not (C.RAW_DIR / "skus.parquet").exists():
        from papertrail.data import generate
        generate.main()
    from papertrail.engine.store import Store
    return Store()
