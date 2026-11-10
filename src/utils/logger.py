"""
One place to set up logging for the whole project.
Change LOG_LEVEL in .env (DEBUG, INFO, WARNING) to see more or less output.
"""

import logging
import os

_configured = False


def get_logger(name):
    global _configured

    if not _configured:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        # hide every HTTP request line from the ollama client
        logging.getLogger("httpx").setLevel(logging.WARNING)
        _configured = True

    return logging.getLogger(name)
