import time
import logging

LOGGER = logging.getLogger("Timer")
logging.basicConfig(level=logging.INFO)

def timeIt(func):
    def wrapper(*args, **kwargs):
        start_time = time.time() * 1000
        data = func(*args, **kwargs)
        end_time = time.time() * 1000
        time_diff = end_time - start_time
        time_in_seconds = time_diff / 1000
        message = func.__name__ + " took " + str(time_in_seconds) + "s"
        LOGGER.warning(message)
        return data
    return wrapper