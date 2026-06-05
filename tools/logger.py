import logging
import sys
    
class ColorFormatter(logging.Formatter):
    ICONS = {
        logging.INFO: "💡",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥",
        logging.DEBUG: "🐛",
    }

    COLORS = {
        logging.DEBUG: "\033[36m",   # cyan
        logging.INFO: "\033[32m",    # green
        logging.WARNING: "\033[33m", # yellow
        logging.ERROR: "\033[31m",   # red
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    PREFIX = {
        logging.DEBUG: "[DEBUG]",   # cyan
        logging.INFO: "[INFO]",    # green
        logging.WARNING: "[WARN]", # yellow
        logging.ERROR: "[ERROR]",   # red
        logging.CRITICAL: "[CRITICAL]",
    }

    def format(self, record):
        color = ColorFormatter.COLORS.get(record.levelno, "")
        icon = ColorFormatter.ICONS.get(record.levelno, "")
        msg = super().format(record)
        if record.levelno == logging.WARNING:
            return f"{icon} {color}{msg}{ColorFormatter.RESET}"
        else:
            return f"{icon}{color}{msg}{ColorFormatter.RESET}"

logging_handler = logging.StreamHandler(stream=sys.stdout)
logging_handler.setFormatter(ColorFormatter("[%(levelname)s] %(message)s"))

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging_handler)