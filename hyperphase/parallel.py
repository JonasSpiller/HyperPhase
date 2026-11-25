"""Process-parallel map over independent SPIDERMAN evaluations."""

from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor


# Process-parallel map over independent SPIDERMAN evaluations (the C extension holds the GIL, so threads give no speedup).
# Workers are recycled afterwards, since SPIDERMAN's allocator growth accumulates in reused workers.
def pmap(fn, items, n_jobs=-1):
    out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(*item) for item in items)
    get_reusable_executor().shutdown(wait=True)
    return out
