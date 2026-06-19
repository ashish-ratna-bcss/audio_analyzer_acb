class ReconciliationError(Exception):
    """A layer emitted fewer units than it received — possible silent drop."""


def check(stage_in: str, count_in, stage_out: str, count_out) -> None:
    if count_out < count_in:
        raise ReconciliationError(
            f"{stage_in}->{stage_out} dropped units: {count_in} -> {count_out}")
