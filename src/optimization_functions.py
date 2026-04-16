"""Hyperparameter optimization utilities for the ML modeling pipeline.

This module provides functions to configure and run RandomizedSearchCV
with custom regression metrics (SMAPE, R², Fugacity SMAPE).
"""

import logging

import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV

from src.train_functions import (build_pipeline,
                                                   fugacity_smape_score,
                                                   smape_score)

logger = logging.getLogger(__name__)


def get_optimization(
    X: np.ndarray,
    y: np.ndarray,
    param_space: dict,
    cv_folds: int,
    n_iter: int,
    seed: int,
) -> RandomizedSearchCV:
    """
    Run a RandomizedSearchCV over CatBoost with SMAPE, R², and Fugacity SMAPE scoring.

    Builds a KFold cross-validator, constructs a CatBoost pipeline, and runs
    a randomized hyperparameter search using three evaluation metrics. The
    search does not refit the best estimator automatically; that step is
    delegated to the caller for manual control over the selection criterion.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    param_space : dict
        Parameter distributions for RandomizedSearchCV. Keys must use the
        ``model__`` prefix (e.g., ``model__depth``).
    cv_folds : int
        Number of folds for KFold cross-validation.
    n_iter : int
        Number of parameter combinations to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object. Results are available via ``search.cv_results_``.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` have incompatible shapes.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import randint
    >>> rng = np.random.default_rng(42)
    >>> X = rng.random((50, 3))
    >>> y = rng.random(50)
    >>> param_space = {"model__depth": randint(4, 7)}
    >>> search = get_optimization(X, y, param_space, cv_folds=3, n_iter=2, seed=42)
    >>> "mean_test_smape" in search.cv_results_
    True
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X={X.shape[0]} and y={y.shape[0]}."
        )

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    smape_scorer = make_scorer(smape_score, greater_is_better=False)
    fugacity_smape_scorer = make_scorer(fugacity_smape_score, greater_is_better=False)

    scoring = {
        "smape": smape_scorer,
        "r2": "r2",
        "fugacity_smape": fugacity_smape_scorer,
    }

    base_pipeline = build_pipeline(CatBoostRegressor(random_seed=seed, verbose=0))

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_space,
        n_iter=n_iter,
        cv=kf,
        scoring=scoring,
        refit=False,
        verbose=10,
        random_state=seed,
        n_jobs=1,  # CatBoost already parallelizes internally
    )

    logger.info("Starting search: %d combinations x %d folds...", n_iter, kf.n_splits)
    search.fit(X, y)
    logger.info("Search completed.")

    return search
