"""SpeakFix: speech transcription and grammar correction for young English learners."""

import os

# Read when NumPy is imported. Its OpenBLAS starts a thread per core and commits about 500 MB for
# them, while the service only does small matrix products. Must stay above any NumPy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "1.0.0"
