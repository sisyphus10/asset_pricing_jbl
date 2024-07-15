"""
Estimates and evaluates a linear asset pricing model.

Computes statistics and saves plots to a folder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data_loading import load_equity_data
from src.estimation_util import camel_case_to_underscore
from src.estimation_util import plot_cumulative_decile_portfolio_returns
from src.estimation_util import plot_equity_curves
from src.estimation_util import set_seeds
from src.evaluation import DatasetStateContainer
from src.models import BaseBetaModel
from src.models import BaseSDFModel
from src.models import BetaModelFromSDFModel
from src.models import BootstrappedLinearSDFModel
from src.models import ElasticNetSDFModel
from src.models import EnsembleFFNBetaModel
from src.models import EnsembleFFNSDFModel
from src.models import LassoCVBetaModel
from src.models import LassoCVSDFModel
from src.models import LinearRegressionBetaModel
from src.models import LinearSDFModel
from src.models import PLSRegressionBetaModel
from src.models import PLSRegressionCVBetaModel
from src.models import RidgeCVBetaModel
from src.models import RidgeSDFModel
from src.models.base import BlockBootstrappedSDFModel
from src.models.base import BootstrappedSDFModel
from src.models.gan_stuff import AdversarialSDF
from src.models.gan_stuff import ConvergenceLossStrategy
from src.models.gan_stuff import EndOnLastEpochStrategy
from src.models.gan_stuff import FixedStepsPerEpochStrategy
from src.models.gan_stuff import TransformerAdversarialSDF

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Logging setup")


set_seeds()


# Store the results in this folder
folder: Path = Path(__file__).parent / "results"
folder.mkdir(exist_ok=True, parents=True)


def main():
    sets, train = build_datasets()

    # Choose the SDF estimation logic here!
    # Shouldn't get worse than 0.2% per epoch
    min_rel_improvement: float = 0.2 / 100.0  # 0.2 %
    patience: int = 20
    print_every_n_steps: int = 10
    n_min_steps: int = 2
    n_epochs: int = 1

    sdf_model = AdversarialSDF(
        print_every_n_steps=print_every_n_steps,
        save_checkpoint=False,
    ).fit(
        train,
        n_epochs=n_epochs,
        sdf_end_strategy=EndOnLastEpochStrategy(
            strategies=[
                ConvergenceLossStrategy(
                    min_rel_improvement=min_rel_improvement,
                    patience=patience,
                ),
                FixedStepsPerEpochStrategy(steps_per_epoch=n_min_steps),
            ],
        ),
        moment_end_strategy=EndOnLastEpochStrategy(
            strategies=[
                ConvergenceLossStrategy(
                    min_rel_improvement=min_rel_improvement,
                    patience=patience,
                ),
                FixedStepsPerEpochStrategy(steps_per_epoch=n_min_steps),
            ],
        ),
    )
    # sdf_model = LinearSDFModel(
    #     double_features=True, subtract_one_half=False, add_constant=False
    # ).fit(train)
    # sdf_model = ElasticNetSDFModel(lambda_1=5e-4, lambda_2=5e-4).fit(train)
    # sdf_model = EnsembleFFNSDFModel(n_models=20, n_epochs=10).fit(train)
    # base_model = RidgeSDFModel()
    # sdf_model = BootstrappedSDFModel(
    #     base_sdf_model=base_model, n_models=100
    # ).fit(train)

    # Try bagged linear SDF model
    # sdf_model = BootstrappedLinearSDFModel().fit(train)
    # sdf_model = LassoCVSDFModel().fit(train)

    _score_sdf_model_on_datasets(sdf_model, sets)

    # Choose the beta estimation model
    # beta_model = LinearRegressionBetaModel().fit(train)
    # beta_model = EnsembleFFNBetaModel(n_models=20, n_epochs=10).fit(train)
    # beta_model = LassoCVBetaModel().fit(train)
    # beta_model = PLSRegressionBetaModel().fit(train)
    # beta_model = RidgeCVBetaModel().fit(train)

    # And run all beta estimators on the SDF model
    for beta_model in (
        LinearRegressionBetaModel(),
        RidgeCVBetaModel(),
        LassoCVBetaModel(),
        PLSRegressionBetaModel(),
        # PLSRegressionCVBetaModel(),
        # EnsembleFFNBetaModel(n_models=20, n_epochs=10),
        BetaModelFromSDFModel(sdf_model),
    ):
        try:
            start_time = pd.Timestamp.now()
            beta_model.fit(train)
            logging.info(
                "\n"
                + 80 * "="
                + "\n"
                + f"\tFitted {beta_model.__class__.__name__} on {sdf_model.__class__.__name__}\n"
                + f"\tTime to fit {beta_model.__class__.__name__}: {pd.Timestamp.now() - start_time}\n"
                + 80 * "=",
            )

            _generate_model_reports(sdf_model, beta_model, sets)

        except Exception as e:
            logging.error(
                f"Failed to fit {beta_model.__class__.__name__} on {sdf_model.__class__.__name__}",
            )
            logging.error(e)

    logging.info("Finished fitting all models!")


def _generate_model_reports(sdf_model, beta_model, sets):
    axs, fig, gs = _setup_plots()
    # Reset the coloring on the charts from matplotlib
    plt.rcParams.update(plt.rcParamsDefault)

    # Add title to the figure
    fig.suptitle(
        f"{sdf_model.__class__.__name__} + {beta_model.__class__.__name__}",
        fontsize=16,
    )

    # Plot the equity curves
    plot_equity_curves(sets, axs=axs[0])
    stats = pd.concat(
        [s.sdf_portfolio_statistics for s in sets],
        axis=1,
        keys=[s.name for s in sets],
    )
    print(stats.round(3))
    pp_dfs = []
    dec = []
    for dataset in sets:
        beta_model.score(dataset)

        pp = dataset.evaluate_pricing_performance()
        pp_dfs.append(pp)
        dec.append(dataset.evaluate_decile_portfolio_performance())
    pp_df = pd.concat(pp_dfs, axis=1, keys=[s.name for s in sets])
    print(pp_df.round(3))
    dec_df = pd.concat(dec, axis=1, keys=[s.name for s in sets])
    print(dec_df.round(3))
    plot_cumulative_decile_portfolio_returns(sets, axs=axs[1])
    # Add the pricing performance table
    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis("off")
    table_data = pp_df.round(3).to_numpy()
    row_labels = pp_df.index
    col_labels = pp_df.columns
    table = ax_table.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    sdf_dir: Path = folder / camel_case_to_underscore(
        sdf_model.__class__.__name__,
    )
    sdf_dir.mkdir(exist_ok=True, parents=True)
    figure_path: Path = sdf_dir / (
        camel_case_to_underscore(beta_model.__class__.__name__) + ".png"
    )
    fig.savefig(figure_path)


def _score_sdf_model_on_datasets(sdf_model, sets):
    for dataset in sets:
        # With the model, estimate the raw sdf weights on all datasets
        sharpe = sdf_model.score(dataset)
        print(f"Sharpe on set {dataset.name}: {sharpe: .4f}")


def build_datasets():
    train_data, train_column_names = load_equity_data(name="train")
    train = DatasetStateContainer(train_data, train_column_names, name="train")

    valid_data, valid_column_names = load_equity_data(name="valid")
    valid = DatasetStateContainer(valid_data, valid_column_names, name="valid")

    test_data, test_column_names = load_equity_data(name="test")
    test = DatasetStateContainer(test_data, test_column_names, name="test")

    sets = [train, valid, test]
    return sets, train


def _setup_plots():
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 4, 2])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    return axs, fig, gs


if __name__ == "__main__":
    main()
