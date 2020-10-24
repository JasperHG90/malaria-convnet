import time
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def log_timing(fn):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        ret = fn(*args, **kwargs)
        t = (time.time() - t1) * 1000
        logging.debug("Function %s took %s ms" % (fn.__name__, t))
        return ret
    return wrapper

