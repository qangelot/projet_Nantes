import logging
import sys

from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic import AnyHttpUrl, BaseSettings


class LoggingSettings(BaseSettings):
    # logging level of the app (CRITICAL, ERROR, WARNING, INFO, DEBUG)
    LOGGING_LEVEL: int = logging.INFO


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    logging: LoggingSettings = LoggingSettings()

    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000", 
        "http://localhost:8000",
        "https://localhost:3000",
        "https://localhost:8000",
    ]

    PROJECT_NAME: str = "Canteen Attendance Prediction API"

    class Config:
        case_sensitive = True


# https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging  # noqa
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # get corresponding loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def setup_app_logging(config: Settings) -> None:
    """Prepare custom logging for the app."""

    # get the default logs from web server (on top of the ones we set up)
    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(
        handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}]
    )

# export settings
settings = Settings()