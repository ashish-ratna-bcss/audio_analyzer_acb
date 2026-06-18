import pytest
from pipeline.reconcile import check, ReconciliationError


def test_equal_counts_ok():
    check("L0", 1, "L1", 1)  # no raise


def test_drop_raises():
    with pytest.raises(ReconciliationError):
        check("L3", 10, "L4", 7)
