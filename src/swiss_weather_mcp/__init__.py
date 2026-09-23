import logging
import logging.config
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("swiss-weather-mcp")
except PackageNotFoundError:
    __version__ = "unknown"


LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure the root logger ("") to capture logs from the entry point, all internal
    sub-modules, and third-party libraries via a single handler writing to stderr.

    Libraries that report their own progress get a fixed level of their own, so raising
    log_level does not bring back the output they would otherwise flood the request with.
    """
    logger_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "stderr_handler": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stderr",  # stdout carries the MCP protocol on the stdio transport
            }
        },
        "loggers": {
            "": {  # "" corresponds to root logger
                "handlers": ["stderr_handler"],
                "level": log_level,
                "propagate": False,
            },
            "httpx": {  # silence httpx HTTP request logs
                "level": "WARNING",
            },
            "urllib3": {  # connection retries
                "level": "WARNING",
            },
            "mcp.server": {  # one line per dispatched MCP request
                "level": "WARNING",
            },
        },
    }
    logging.config.dictConfig(logger_config)


__all__ = ["__version__", "LOG_LEVELS", "setup_logging"]
