# captcha/logger.py

from loguru import logger
import sys
from pathlib import Path
import pytorch_lightning as pl
# Ensure the logs directory exists
Path("outputs").mkdir(parents=True, exist_ok=True)

# Remove the default Loguru handler to prevent duplicate logs
logger.remove()

# Add a console handler with a specific format and INFO level
logger.add(
    sys.stdout, 
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", 
    level="INFO",
    colorize=True
)


# Export the logger for use in other modules
__all__ = ["logger"]


