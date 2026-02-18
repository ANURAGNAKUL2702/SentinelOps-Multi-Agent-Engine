"""Database connection management — engine, session factory, pooling."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..config import ReportingConfig
from ..telemetry import get_logger

_logger = get_logger(__name__)


class DatabaseConnection:
    """Manage SQLAlchemy engine and session factory.

    Args:
        config: Reporting configuration (uses ``database_url``).
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()
        url = self.config.database_url

        engine_kwargs: dict = {}
        if url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                engine_kwargs["poolclass"] = StaticPool
        else:
            engine_kwargs["pool_size"] = 5
            engine_kwargs["max_overflow"] = 10
            engine_kwargs["pool_pre_ping"] = True

        self._engine: Engine = create_engine(url, **engine_kwargs)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    @property
    def engine(self) -> Engine:
        """Return the SQLAlchemy engine."""
        return self._engine

    def create_tables(self) -> None:
        """Create all ORM tables using :data:`Base.metadata`.

        This is a convenience for development/test; in production
        use Alembic migrations.
        """
        from .models import Base

        Base.metadata.create_all(self._engine)

    def drop_tables(self) -> None:
        """Drop all ORM tables (test cleanup)."""
        from .models import Base

        Base.metadata.drop_all(self._engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager yielding a :class:`Session`.

        Commits on success, rolls back on exception, and always closes.
        """
        sess: Session = self._session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    def close(self) -> None:
        """Dispose the engine and release connection pool."""
        self._engine.dispose()
        _logger.info("Database connection closed")
