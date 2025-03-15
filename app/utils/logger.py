"""
This Module is responsible for Managing
all of the logging actions

Usage:
    from app.utils.logger import LogHandler

    logger = LogHandler.get_logger()

    logger.info("This is an info message")

    logger.error("This is an error message")

    logger.exception("This is an exception message")
"""

import sys
import logging
from setuptools._distutils.util import strtobool
from loguru import logger
from app.utils.config import settings


class LogHandler:
    """
    This class is responsible for managing the logger.
    """

    _logger = None

    @classmethod
    def get_logger(cls, log_file_path: str = settings.LOG_FILE):
        """
        Gets the logger instance.

        Args:
            log_file_path (str): The path to the log file. Defaults to `settings.LOG_FILE`.

        Returns:
            loguru.Logger: The logger instance.

        Notes:
            The logger is a singleton, meaning that it is only initialized once.
            The log format is set to "{time} {level} {message}" and the log level is set to INFO.
            The log is also serialized to JSON.
            Backtraces are enabled, which will show full stack traces for errors.
            The log is compressed to a zip file when it reaches 100 MB in size.
            Optionally, if `settings.SQL_LOG` is set to True, SQLAlchemy logging is enabled.
        """
        if cls._logger is None:
            # Initialize the logger only once (singleton)
            cls._logger = logger
            cls._logger.remove()

            # Log to the console
            cls._logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

            # Log to a file with custom settings
            cls._logger.add(
                log_file_path,
                format="{time} {level} {message}",  # log format
                serialize=True,  # Enable JSON serialization
                level="INFO",  # Set log level to INFO
                backtrace=True,  # Enable backtrace to show full stack traces
                compression="zip",  # Compress old log files as .zip
                rotation="100 MB",  # Rotate the log when it reaches 100 MB
            )

            # if bool(strtobool(settings.SQL_LOG)):
            #     # Optionally, set up SQLAlchemy logging
            #     cls._setup_sqlalchemy_logging()
        return cls._logger

    # @staticmethod
    # def _setup_sqlalchemy_logging():
        """Redirect SQLAlchemy logs to Loguru"""

        class LoguruHandler(logging.Handler):
            """
            A custom logging handler that redirects SQLAlchemy logs to Loguru.

            This handler is used to capture SQLAlchemy logs and redirect them to the Loguru logger.
            """

            def emit(self, record):
                """
                Redirects a logging record to Loguru.

                This method is called by the logging system for each log record. It redirects
                the log record to the Loguru logger, which is configured to log to the console
                and a file.

                :param record: The log record to redirect.
                :type record: logging.LogRecord
                """
                logger_opt = logger.opt(depth=6, exception=record.exc_info)
                logger_opt.log(record.levelno, record.getMessage())

        sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
        sqlalchemy_logger.setLevel(logging.INFO)
        sqlalchemy_logger.addHandler(LoguruHandler())