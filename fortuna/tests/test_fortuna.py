"""
Unit and regression test for the fortuna package.
"""

# Import package, test suite, and other packages as needed
import fortuna
import pytest
import sys

def test_fortuna_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "fortuna" in sys.modules
