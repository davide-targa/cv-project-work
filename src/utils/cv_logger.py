import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "standard": {"formatter": "standard", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"},
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["standard"],
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pf")
