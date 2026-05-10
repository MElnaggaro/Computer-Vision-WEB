import os
import pytest

# Prevent libiomp5md.dll initialization error / Access Violation on Windows
# when importing PyTorch and OpenCV in the same process.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Disable tqdm threading / monitoring to prevent access violation on exit
os.environ["TQDM_DISABLE"] = "1"

@pytest.fixture(autouse=True)
def setup_test_env():
    # Force some extra stability vars just in case
    pass
