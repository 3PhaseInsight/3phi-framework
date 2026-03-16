import logging
from collections.abc import Iterable

from sqlalchemy import insert
from sqlalchemy.orm import Session


class BaseResource:
    def __init__(self, session: Session):
        self.s = session
        self.logger = logging.getLogger(self.__class__.__name__)

    def _log_debug(self, msg: str, **extra):
        self.logger.debug(msg, extra=extra)

    def _log_info(self, msg: str, **extra):
        self.logger.info(msg, extra=extra)

    def _log_warning(self, msg: str, **extra):
        self.logger.warning(msg, extra=extra)

    def _log_error(self, msg: str, **extra):
        self.logger.error(msg, extra=extra)

    def bulk_insert(self, table, rows: Iterable[dict]):
        rows = list(rows)
        if not rows:
            return 0
        self.s.execute(insert(table).values(rows))
        return len(rows)
