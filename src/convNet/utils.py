import time


def log_timing(logger):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            ret = fn(*args, **kwargs)
            t = (time.time() - t1) * 1000
            logger.debug("Function %s took %s ms" % (fn.__name__, t))
            return ret
        return wrapper
    return decorator

