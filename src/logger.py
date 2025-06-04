import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger object
logger = logging.getLogger('OpenAlitaLogger')

def log_info(message: str) -> None:
    """Log an informational message."""
    logger.info(message)

def log_warning(message: str) -> None:
    """Log a warning message."""
    logger.warning(message)

def log_error(message: str) -> None:
    """Log an error message."""
    logger.error(message)

def log_debug(message: str) -> None:
    """Log a debug message."""
    logger.debug(message)