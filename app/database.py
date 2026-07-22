from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .settings import settings
from .logging_utils import log_call

engine = create_engine(
    settings.sqlite_url,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


@log_call
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
