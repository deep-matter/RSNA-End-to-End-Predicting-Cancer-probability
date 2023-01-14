import platform, sys, pydicom

print(
    platform.platform(),
    "\nPython", sys.version,
    "\npydicom", pydicom.__version__
)

try:
    import pylibjpeg
    
    from pylibjpeg.utils import get_decoders
    print(pylibjpeg.__version__, get_decoders())
except ImportError:
    print("pylibjpeg not found")

try:
    import libjpeg
    print(libjpeg.__version__)
except ImportError:
    print("libjpeg not found")