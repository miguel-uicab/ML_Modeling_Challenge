import math

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.nonparametric.smoothers_lowess import lowess


def get_spearman_matrix(df: pl.DataFrame) -> None:
    """
    Compute and display a Spearman correlation heatmap for a Polars DataFrame.

    Calculates the Spearman rank-correlation matrix across all columns in the
    input DataFrame and renders an annotated heatmap using matplotlib.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing numeric columns. All columns are included
        in the correlation computation.

    Returns
    -------
    None
        Displays the heatmap inline via matplotlib.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.read_csv("data/training_data.csv")
    >>> get_spearman_matrix(df)
    """
    data = df.to_numpy()
    cols = df.columns
    corr_matrix, _ = spearmanr(data)

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlación de Spearman")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
            )

    ax.set_title("Matriz de correlación de Spearman — df_train", fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()


def get_mutual_information(
    df: pl.DataFrame,
    target_col: str = "target",
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Compute mutual information scores between features and a target column.

    Estimates the mutual information (MI) between each feature and the target
    using a k-nearest-neighbor approach, then displays a bar chart ranked by
    MI score. Returns a DataFrame with the scores for downstream filtering.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing feature columns and the target column.
    target_col : str, optional
        Name of the target column, by default "target".
    random_state : int, optional
        Random seed for reproducibility of the MI estimator, by default 42.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``feature`` and ``mutual_info``, sorted
        descending by mutual information score.

    Raises
    ------
    ValueError
        If ``target_col`` is not found in ``df``.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.read_csv("data/training_data.csv")
    >>> mi_df = get_mutual_information(df)
    >>> mi_df.head()
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    feature_cols = [c for c in df.columns if c != target_col]
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()

    mi_scores = mutual_info_regression(X, y, random_state=random_state)

    mi_df = pl.DataFrame({"feature": feature_cols, "mutual_info": mi_scores}).sort(
        "mutual_info", descending=True
    )

    fig = px.bar(
        mi_df,
        x="feature",
        y="mutual_info",
        text_auto=".3f",
        title=f"Dependencia con '{target_col}' (Información mutua)",
        labels={"mutual_info": "Información mutua", "feature": "Feature"},
        color="mutual_info",
        color_continuous_scale="Viridis",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=550)
    fig.show()

    return mi_df


def get_scatter_plots(
    df: pl.DataFrame,
    features: list[str],
    target_col: str = "target",
    n_cols: int = 3,
    lowess_frac: float = 0.3,
) -> None:
    """
    Display scatter plots of selected features against a target column with LOWESS trend lines.

    For each feature in the provided list, renders a scatter plot against the
    target column and overlays a LOWESS smoothing curve to reveal non-linear
    relationships.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the feature and target columns.
    features : list[str]
        List of feature column names to plot.
    target_col : str, optional
        Name of the target column, by default "target".
    n_cols : int, optional
        Number of columns in the subplot grid, by default 3.
    lowess_frac : float, optional
        Fraction of data used for each LOWESS estimate. Smaller values produce
        a less smooth curve, by default 0.3.

    Returns
    -------
    None
        Displays the subplot grid inline via Plotly.

    Raises
    ------
    ValueError
        If ``target_col`` is not found in ``df``.
    ValueError
        If any feature in ``features`` is not found in ``df``.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.read_csv("data/training_data.csv")
    >>> get_scatter_plots(df, features=["feature_2", "feature_13"])
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features not found in DataFrame: {missing}")

    n_rows = math.ceil(len(features) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)

    for i, feature in enumerate(features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        x = df[feature].to_numpy()
        y = df[target_col].to_numpy()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=4, opacity=0.4),
                name=feature,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # LOWESS trend line captures non-linear relationships
        smoothed = lowess(y, x, frac=lowess_frac, return_sorted=True)
        fig.add_trace(
            go.Scatter(
                x=smoothed[:, 0],
                y=smoothed[:, 1],
                mode="lines",
                line=dict(color="red", width=2),
                name=f"{feature} lowess",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text=feature, row=row, col=col)
        fig.update_yaxes(title_text=target_col if col == 1 else "", row=row, col=col)

    fig.update_layout(
        title_text=f"Scatter Plots: {target_col} vs features (LOWESS)",
        height=350 * n_rows,
        width=1200,
    )
    fig.show()


def get_histograms(
    df: pl.DataFrame,
    features: list[str],
    target_col: str = "target",
    n_cols: int = 3,
    n_bins: int = 40,
) -> None:
    """
    Display boxplot and histogram panels for selected features and the target column.

    For each column (features + target), renders a paired boxplot above a histogram
    to simultaneously visualize the spread, central tendency, and distribution shape.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the feature and target columns.
    features : list[str]
        List of feature column names to include in the plot.
    target_col : str, optional
        Name of the target column appended at the end of the panels, by default "target".
    n_cols : int, optional
        Number of columns in the subplot grid, by default 3.
    n_bins : int, optional
        Number of bins for each histogram, by default 40.

    Returns
    -------
    None
        Displays the subplot grid inline via Plotly.

    Raises
    ------
    ValueError
        If ``target_col`` is not found in ``df``.
    ValueError
        If any feature in ``features`` is not found in ``df``.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.read_csv("data/training_data.csv")
    >>> get_histograms(df, features=["feature_2", "feature_13"])
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features not found in DataFrame: {missing}")

    all_cols = features + [target_col]
    n_rows = math.ceil(len(all_cols) / n_cols)

    fig = make_subplots(
        rows=n_rows * 2,
        cols=n_cols,
        subplot_titles=[c for c in all_cols for _ in range(2)][::2]
        + [""] * (n_rows * n_cols - len(all_cols)),
        row_heights=[0.25, 0.75] * n_rows,
        vertical_spacing=0.04,
        horizontal_spacing=0.08,
    )

    for i, col_name in enumerate(all_cols):
        grid_row = i // n_cols
        grid_col = i % n_cols + 1
        box_row = grid_row * 2 + 1
        hist_row = grid_row * 2 + 2

        values = df[col_name].to_numpy()

        fig.add_trace(
            go.Box(
                x=values,
                marker_color="#636EFA",
                showlegend=False,
                name=col_name,
                boxmean=True,
            ),
            row=box_row,
            col=grid_col,
        )

        fig.add_trace(
            go.Histogram(
                x=values,
                marker_color="#636EFA",
                opacity=0.75,
                showlegend=False,
                name=col_name,
                nbinsx=n_bins,
            ),
            row=hist_row,
            col=grid_col,
        )

        # Hide boxplot x-axis ticks to align with histogram
        fig.update_xaxes(showticklabels=False, row=box_row, col=grid_col)
        fig.update_yaxes(showticklabels=False, row=box_row, col=grid_col)
        fig.update_xaxes(title_text=col_name, row=hist_row, col=grid_col)

    fig.update_layout(
        title_text="Distribución de features y target (Boxplot + Histograma)",
        height=280 * n_rows * 2,
        width=1200,
        bargap=0.05,
    )
    fig.show()


def get_train_test_comparison(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    features: list[str],
    n_cols: int = 3,
    n_bins: int = 40,
) -> None:
    """
    Compare the distribution of features between train and test sets via overlapping histograms.

    For each feature, renders overlapping normalized histograms for both datasets
    to detect covariate shift — distributional differences that could hurt
    out-of-sample generalization.

    Parameters
    ----------
    df_train : pl.DataFrame
        Training DataFrame containing the feature columns.
    df_test : pl.DataFrame
        Test DataFrame containing the feature columns.
    features : list[str]
        List of feature column names to compare.
    n_cols : int, optional
        Number of columns in the subplot grid, by default 3.
    n_bins : int, optional
        Number of bins for each histogram, by default 40.

    Returns
    -------
    None
        Displays the subplot grid inline via Plotly.

    Raises
    ------
    ValueError
        If any feature in ``features`` is not found in ``df_train`` or ``df_test``.

    Examples
    --------
    >>> import polars as pl
    >>> df_train = pl.read_csv("data/training_data.csv")
    >>> df_test = pl.read_csv("data/blind_test_data.csv")
    >>> get_train_test_comparison(df_train, df_test, features=["feature_2", "feature_13"])
    """
    missing_train = [f for f in features if f not in df_train.columns]
    missing_test = [f for f in features if f not in df_test.columns]
    if missing_train:
        raise ValueError(f"Features not found in df_train: {missing_train}")
    if missing_test:
        raise ValueError(f"Features not found in df_test: {missing_test}")

    n_rows = math.ceil(len(features) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)

    for i, feature in enumerate(features):
        row = i // n_cols + 1
        col = i % n_cols + 1

        fig.add_trace(
            go.Histogram(
                x=df_train[feature].to_numpy(),
                name="Train",
                nbinsx=n_bins,
                opacity=0.6,
                marker_color="#636EFA",
                histnorm="probability density",
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Histogram(
                x=df_test[feature].to_numpy(),
                name="Test",
                nbinsx=n_bins,
                opacity=0.6,
                marker_color="#EF553B",
                histnorm="probability density",
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text=feature, row=row, col=col)

    fig.update_layout(
        title_text="Comparación de distribuciones Train vs Test",
        barmode="overlay",
        height=350 * n_rows,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
