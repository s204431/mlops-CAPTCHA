from loguru import logger
import sys

# Remove the default logger
logger.remove() 

# Add a new logger with a custom format
logger.add(
    sys.stdout, 
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", 
    level="INFO",
    colorize=True
)

# Export the logger for use in other modules
__all__ = ["logger"]


