import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
from petal import hello


def test_hello_expected() -> None:
    assert hello("Jeff") == "Hello, Jeff!"


def test_hello_empty() -> None:
    with pytest.raises(ValueError):
        hello("")


def test_hello_edge_case() -> None:
    assert hello("A") == "Hello, A!"
