import json
import logging
import logging.config
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

try:
    __version__ = version("meteo-swiss-mcp")
except PackageNotFoundError:
    __version__ = "unknown"


def setup_logging() -> None:
    try:
        config = json.loads((files(__name__) / "log_config.json").read_text())
        logging.config.dictConfig(config)
        logging.getLogger(__name__).info("Logger successfully configured")
    except Exception as e:
        # basicConfig() attaches the root logger to STDERR, so it cannot interfere with the stdio MCP protocol
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error(f"Failed to configure logger: {e}", exc_info=True)


__all__ = ["__version__", "setup_logging"]
