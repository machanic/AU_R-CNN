import contextlib
import ctypes
from ctypes.util import find_library

# Prioritize hand-compiled OpenBLAS library over version in /usr/lib/
# from Ubuntu repos
try_paths = ['/usr/local/anaconda3/lib/libopenblas.so',
             '/usr/local/anaconda3/lib/libopenblas_power8p-r0.2.19.so',
             '/usr/local/anaconda3/lib/libopenblas.so.0',
             find_library('openblas')]
openblas_lib = None
for libpath in try_paths:
    try:
        openblas_lib = ctypes.cdll.LoadLibrary(libpath)
        break
    except OSError:
        continue
if openblas_lib is None:
    raise EnvironmentError('Could not locate an OpenBLAS shared library', 2)


def set_num_threads(n):
    """Set the current number of threads used by the OpenBLAS server."""
    openblas_lib.openblas_set_num_threads(int(n))


# At the time of writing these symbols were very new:
# https://github.com/xianyi/OpenBLAS/commit/65a847c
try:
    openblas_lib.openblas_get_num_threads()
    def get_num_threads():
        """Get the current number of threads used by the OpenBLAS server."""
        return openblas_lib.openblas_get_num_threads()
except AttributeError:
    def get_num_threads():
        """Dummy function (symbol not present in %s), returns -1."""
        return -1
    pass

try:
    openblas_lib.openblas_get_num_procs()
    def get_num_procs():
        """Get the total number of physical processors"""
        return openblas_lib.openblas_get_num_procs()
except AttributeError:
    def get_num_procs():
        """Dummy function (symbol not present), returns -1."""
        return -1
    pass


@contextlib.contextmanager
def num_threads(n):
    """Temporarily changes the number of OpenBLAS threads.

    Example usage:

        print("Before: {}".format(get_num_threads()))
        with num_threads(n):
            print("In thread context: {}".format(get_num_threads()))
        print("After: {}".format(get_num_threads()))
    """
    old_n = get_num_threads()
    set_num_threads(n)
    try:
        yield
    finally:
        set_num_threads(old_n)
