import json
import logging
import logging.config
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

try:
    __version__ = version("meteo-swiss-mcp")
except PackageNotFoundError:
    __version__ = "unknown"


LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging from the packaged log_config.json.

    The per-library levels in that file stay as they are, only the handler and
    the root logger follow log_level, so raising verbosity does not bring back
    the third-party output the config deliberately quiets.
    """
    try:
        config = json.loads((files(__name__) / "log_config.json").read_text())
        config["handlers"]["stderr"]["level"] = log_level
        config["root"]["level"] = log_level
        logging.config.dictConfig(config)
        logging.getLogger(__name__).debug("Logger successfully configured")
    except Exception as e:
        # basicConfig() attaches the root logger to STDERR, so it cannot interfere with the stdio MCP protocol
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error(f"Failed to configure logger: {e}", exc_info=True)


__all__ = ["__version__", "LOG_LEVELS", "setup_logging"]
