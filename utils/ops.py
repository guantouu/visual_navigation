import math

def log_uniform(lo, hi, rate):
    """
        draw a sample from a log-uniform distribution in [lo, hi]
    """
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1-rate) + log_hi * rate
    return math.exp(v)
def sample_action():
    pass