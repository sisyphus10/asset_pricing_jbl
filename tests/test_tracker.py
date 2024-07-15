import pytest

from src.tracker import RunTracker


def test_tracker(tmp_path):
    tracker = RunTracker(tmp_path)

    tracker.set_csv_columns(
        columns=[
            "loss",
            "val_loss",
            "new_column",
        ]
    )
    tracker.save_losses({"loss": 0.1, "val_loss": 0.2, "new_column": 12.0})
    tracker.save_losses(
        {
            "loss": 0.3,
        }
    )
    tracker.save_losses(
        {
            "new_column": 15.0,
            "loss": 0.6,
        }
    )

    assert (tmp_path / "losses.csv").exists()
