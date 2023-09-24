import logging
import os
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
from logging.handlers import RotatingFileHandler

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)


# Create and configure your custom logger
logger = logging.getLogger("clearvision")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevents log messages from being passed to the root logger

# Add handlers to your custom logger
file_handler = RotatingFileHandler(
    "clearvision.log", maxBytes=1e6, backupCount=3
)  # 1 MB file size limit, keep last 3 logs
stream_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "clearvision"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
