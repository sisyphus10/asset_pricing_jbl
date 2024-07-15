from __future__ import annotations

__version__ = "0.1.0"

from datetime import datetime
from logging import getLogger
from pathlib import Path

import pandas as pd

now_formatted = datetime.now().strftime("%Y%m%d_%H%M%S")

project_dir = (
    Path.home() / "research/pricing_kernels" / __version__ / now_formatted
)
project_dir.mkdir(parents=True, exist_ok=True)

models_dir = project_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

stats_dir = project_dir / "stats"
stats_dir.mkdir(parents=True, exist_ok=True)

logger = getLogger(__name__)
logger.info(f"Created project directory at {project_dir}")
logger.info(f"Created models directory at {models_dir}")
logger.info(f"Created stats directory at {stats_dir}")


def save_training_stats(stats: pd.DataFrame, name: str):
    stats.to_csv(stats_dir / f"{name}.csv")
    logger.info(f"Saved training stats to {stats_dir / name}.csv")


chkpt_id: int = 0


def save_model_checkpoint(model, name: str):
    global chkpt_id
    model.save(models_dir / f"{name}_{chkpt_id:03d}.h5")
    logger.info(f"Saved model checkpoint to {models_dir / name}.h5")
    chkpt_id += 1
