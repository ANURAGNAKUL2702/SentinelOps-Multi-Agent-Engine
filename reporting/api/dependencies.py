"""Shared FastAPI dependencies — singletons & session factories."""

from __future__ import annotations

from typing import Optional

from ..config import ReportingConfig
from ..database.connection import DatabaseConnection
from ..database.repository import IncidentRepository
from ..report_builder import ReportBuilder

# Module-level singletons (initialised at startup)
_config: Optional[ReportingConfig] = None
_db_conn: Optional[DatabaseConnection] = None
_repository: Optional[IncidentRepository] = None
_report_builder: Optional[ReportBuilder] = None


def init_dependencies(config: Optional[ReportingConfig] = None) -> None:
    """Initialise shared singletons.  Called once during app startup."""
    global _config, _db_conn, _repository, _report_builder  # noqa: PLW0603
    _config = config or ReportingConfig()
    _db_conn = DatabaseConnection(_config)
    _db_conn.create_tables()
    _repository = IncidentRepository(_db_conn)
    _report_builder = ReportBuilder(_config)


def shutdown_dependencies() -> None:
    """Cleanup on shutdown."""
    global _db_conn  # noqa: PLW0603
    _db_conn = None


def get_config() -> ReportingConfig:
    """Return the shared :class:`ReportingConfig`."""
    if _config is None:
        init_dependencies()
    return _config  # type: ignore[return-value]


def get_db() -> DatabaseConnection:
    """Return the shared :class:`DatabaseConnection`."""
    if _db_conn is None:
        init_dependencies()
    return _db_conn  # type: ignore[return-value]


def get_repository() -> IncidentRepository:
    """Return the shared :class:`IncidentRepository`."""
    if _repository is None:
        init_dependencies()
    return _repository  # type: ignore[return-value]


def get_report_builder() -> ReportBuilder:
    """Return the shared :class:`ReportBuilder`."""
    if _report_builder is None:
        init_dependencies()
    return _report_builder  # type: ignore[return-value]
