import sys
import pytest
from pathlib import Path

# Automatically add backend directory to sys.path for pytest and test runners
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
FIXTURES_DIR = BACKEND_DIR / "tests" / "fixtures"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

@pytest.fixture(scope="session")
def backend_path():
    return BACKEND_DIR

@pytest.fixture(scope="session")
def repo_root_path():
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def fixtures_path():
    return FIXTURES_DIR
