import os
from functools import lru_cache

from sqlalchemy import URL, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# DB_TYPES = {
#     "postgres": "POSTGRES",
#     "microsoft_sql_server": "MS_SQL"
# }


def db_url() -> str | URL:
    # db_type = os.getenv("DB_TYPE", DB_TYPES["postgres"])
    # if db_type == DB_TYPES["microsoft_sql_server"]:
    #     # using Microsoft SQL Server
    #     return URL.create(
    #         drivername="mssql+pyodbc",
    #         username=os.environ["DB_USER"],
    #         password=os.environ["DB_PASSWORD"],
    #         host=os.environ["DB_HOST"],
    #         port=int(os.environ["DB_PORT"]),
    #         database=os.environ["DB_NAME"]
    #     )
    # else:
    # default case: POSTGRES
    return (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:"
        f"{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:"
        f"{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    )


@lru_cache(maxsize=1)
def get_engine():
    """
    Lazily create one Engine per process. Dask workers will each have their own.
    For multi-process / distributed, QueuePool is fine, but keep it small.
    Use NullPool if workers are extremely short-lived or if you see FD leaks.
    """
    return create_engine(
        db_url(),
        poolclass=QueuePool,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "5")),
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )


@lru_cache(maxsize=1)
def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False, class_=Session)


def new_session() -> Session:
    """
    Minimal-refactor helper: returns a NEW Session each call.
    Safe for distributed workers as long as you create/use it INSIDE the task.
    """
    return get_session_factory()()
