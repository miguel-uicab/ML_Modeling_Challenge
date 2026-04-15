"""Training utility functions for the ML modeling pipeline.

This module provides reusable building blocks for pipeline construction,
cross-validation, metric computation, and feature importance extraction
for regression models.
"""

import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

SEED = 5000


def smape_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE is a percentage-based error metric symmetric around zero,
    bounded between 0 and 200.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        SMAPE value as a percentage (0–200 range).

    Examples
    --------
    >>> import numpy as np
    >>> smape_score(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    10.909090909090907
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(numerator / denominator) * 100)


def mape_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE), skipping zero targets.

    Zero-valued targets are excluded from the computation to avoid
    division by zero.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        MAPE value as a percentage.

    Examples
    --------
    >>> import numpy as np
    >>> mape_score(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    10.0
    """
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def fugacity_mape_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.15,
) -> float:
    """
    Compute the Fugacity-MAPE metric: percentage of predictions whose individual
    absolute percentage error (MAPE) exceeds a given threshold.

    Measures the proportion of samples where the individual MAPE is above
    ``threshold``. Useful for penalising models that produce a high rate of
    large outlier predictions. Zero-valued targets are excluded to avoid
    division by zero.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).
    threshold : float, optional
        Individual-MAPE threshold expressed as a ratio (e.g. 0.15 = 15%),
        by default 0.15.

    Returns
    -------
    float
        Percentage of non-zero-target samples whose individual MAPE exceeds
        ``threshold`` (value in [0, 100]).

    Examples
    --------
    >>> import numpy as np
    >>> fugacity_mape_score(np.array([10.0, 20.0, 30.0]), np.array([12.0, 20.0, 30.0]))
    33.33333333333333
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    mape_individuals = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.sum(mape_individuals > threshold) / len(mape_individuals) * 100)


def fugacity_smape_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.15,
) -> float:
    """
    Compute the Fugacity-SMAPE metric: percentage of predictions whose individual
    symmetric absolute percentage error (SMAPE) exceeds a given threshold.

    Analogous to ``fugacity_mape_score`` but uses the symmetric formulation
    so both numerator and denominator are affected by the predicted value,
    making the metric more robust to near-zero targets.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).
    threshold : float, optional
        Individual-SMAPE threshold expressed as a ratio (e.g. 0.15 = 15%),
        by default 0.15.

    Returns
    -------
    float
        Percentage of samples whose individual SMAPE exceeds ``threshold``
        (value in [0, 100]).

    Examples
    --------
    >>> import numpy as np
    >>> fugacity_smape_score(np.array([10.0, 20.0, 30.0]), np.array([12.0, 20.0, 30.0]))
    33.33333333333333
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    smape_individuals = np.abs(y_true - y_pred) / (
        (np.abs(y_true) + np.abs(y_pred)) / 2
    )
    return float(np.sum(smape_individuals > threshold) / len(smape_individuals) * 100)


def build_pipeline(estimator: object) -> Pipeline:
    """
    Build a MinMaxScaler → estimator sklearn Pipeline.

    Features are scaled to the [0, 1] range before being passed to
    the estimator, ensuring scale-invariant training.

    Parameters
    ----------
    estimator : object
        A scikit-learn–compatible regressor implementing fit/predict.

    Returns
    -------
    Pipeline
        Unfitted Pipeline with steps ``("scaler", MinMaxScaler())``
        and ``("model", estimator)``.

    Examples
    --------
    >>> from sklearn.linear_model import ElasticNet
    >>> pipe = build_pipeline(ElasticNet())
    >>> list(pipe.named_steps.keys())
    ['scaler', 'model']
    """
    return Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("model", estimator),
        ]
    )


def run_cross_validation(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = SEED,
) -> dict[str, float]:
    """
    Run shuffled k-fold cross-validation and return averaged regression metrics.

    Computes RMSE, MAE, MAPE, SMAPE, and R² across all folds. The pipeline
    is cloned before each fold to ensure independence between folds.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn Pipeline to evaluate.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target array of shape (n_samples,).
    n_splits : int, optional
        Number of CV folds, by default 5.
    seed : int, optional
        Random seed for fold shuffle reproducibility, by default 5000.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``rmse_cv``, ``mae_cv``, ``mape_cv``,
        ``smape_cv``, ``r2_cv``, ``fugacity_mape_cv``, and
        ``fugacity_smape_cv``, each averaged across all folds.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import ElasticNet
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 5))
    >>> y = rng.random(100)
    >>> pipe = build_pipeline(ElasticNet())
    >>> metrics = run_cross_validation(pipe, X, y)
    >>> list(metrics.keys())
    ['rmse_cv', 'mae_cv', 'mape_cv', 'smape_cv', 'r2_cv', 'fugacity_mape_cv', 'fugacity_smape_cv']
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    smape_scorer = make_scorer(smape_score, greater_is_better=False)
    mape_scorer = make_scorer(mape_score, greater_is_better=False)
    fugacity_mape_scorer = make_scorer(fugacity_mape_score, greater_is_better=False)
    fugacity_smape_scorer = make_scorer(fugacity_smape_score, greater_is_better=False)

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "mape": mape_scorer,
        "smape": smape_scorer,
        "r2": "r2",
        "fugacity_mape": fugacity_mape_scorer,
        "fugacity_smape": fugacity_smape_scorer,
    }

    cv_raw = cross_validate(pipeline, X, y, cv=kf, scoring=scoring)

    return {
        "rmse_cv": float(-np.mean(cv_raw["test_rmse"])),
        "mae_cv": float(-np.mean(cv_raw["test_mae"])),
        "mape_cv": float(-np.mean(cv_raw["test_mape"])),
        "smape_cv": float(-np.mean(cv_raw["test_smape"])),
        "r2_cv": float(np.mean(cv_raw["test_r2"])),
        "fugacity_mape_cv": float(-np.mean(cv_raw["test_fugacity_mape"])),
        "fugacity_smape_cv": float(-np.mean(cv_raw["test_fugacity_smape"])),
    }


def get_feature_importances(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pl.DataFrame:
    """
    Extract feature importances or absolute coefficients from a fitted pipeline.

    Handles tree-based models exposing ``feature_importances_`` and linear
    models exposing ``coef_``. Returns results sorted by importance descending.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline containing a ``"model"`` step.
    feature_names : list[str]
        Names of the input features in the order used during training.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns ``feature`` and ``importance``,
        sorted in descending order by importance.

    Raises
    ------
    ValueError
        If the model step exposes neither ``feature_importances_`` nor
        ``coef_``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> rng = np.random.default_rng(0)
    >>> pipe = build_pipeline(RandomForestRegressor(random_state=42))
    >>> pipe.fit(rng.random((50, 3)), rng.random(50))
    Pipeline(...)
    >>> df = get_feature_importances(pipe, ["a", "b", "c"])
    >>> df.columns
    ['feature', 'importance']
    """
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances: np.ndarray = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        raise ValueError(
            f"Model {type(model).__name__} does not expose "
            "feature_importances_ or coef_."
        )

    return pl.DataFrame(
        {"feature": feature_names, "importance": importances.tolist()}
    ).sort("importance", descending=True)


def get_model_pipelines(seed: int = SEED) -> dict[str, Pipeline]:
    """
    Build a dictionary of named, unfitted sklearn Pipelines for regression.

    Each pipeline contains a MinMaxScaler followed by a regressor.
    All stochastic estimators are initialised with the provided seed to
    guarantee reproducibility across runs.

    Models included:
    - Lasso
    - DecisionTree
    - RandomForest
    - GradientBoosting
    - XGBoost
    - LightGBM
    - CatBoost

    Parameters
    ----------
    seed : int, optional
        Random state passed to every estimator that accepts it,
        by default 5000.

    Returns
    -------
    dict[str, Pipeline]
        Mapping of model name to its unfitted Pipeline.

    Examples
    --------
    >>> pipelines = get_model_pipelines(seed=42)
    >>> list(pipelines.keys())
    ['Lasso', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
    >>> pipelines["RandomForest"].named_steps["model"].random_state
    42
    """
    return {
        "Lasso": build_pipeline(Lasso(random_state=seed)),
        "DecisionTree": build_pipeline(DecisionTreeRegressor(random_state=seed)),
        "RandomForest": build_pipeline(
            RandomForestRegressor(random_state=seed, n_jobs=-1)
        ),
        "GradientBoosting": build_pipeline(
            GradientBoostingRegressor(random_state=seed)
        ),
        "XGBoost": build_pipeline(
            XGBRegressor(
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
        ),
        "LightGBM": build_pipeline(
            LGBMRegressor(
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
        ),
        "CatBoost": build_pipeline(
            CatBoostRegressor(
                random_seed=seed,
                verbose=0,
            )
        ),
    }
