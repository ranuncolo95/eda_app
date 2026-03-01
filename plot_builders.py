import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_histogram(
    df: pd.DataFrame,
    x: str,
    nbins: int = 30,
    color: str | None = None,
    opacity: float = 0.75,
    histnorm: str | None = None,
    marginal_mode: str | None = "box",   # None | "box" | "violin" | "rug"
    barmode: str = "overlay",            # "overlay" | "group" | "relative" | "stack"
    color_discrete_map: dict[str, str] | None = None,
):
    allowed_marginals = {None, "box", "violin", "rug"}
    if marginal_mode not in allowed_marginals:
        marginal_mode = "box"

    allowed_barmodes = {"overlay", "group", "relative", "stack"}
    if barmode not in allowed_barmodes:
        barmode = "overlay"

    plot_df = df.copy()

    # Per rendere affidabile la mappa colori con categorie/missing
    if color is not None and color_discrete_map is not None:
        plot_df[color] = plot_df[color].astype("string").fillna("<NA>")

    fig = px.histogram(
        plot_df,
        x=x,
        nbins=nbins,
        color=color,
        marginal=marginal_mode,
        histnorm=histnorm,
        color_discrete_map=color_discrete_map,
    )

    if color is not None:
        fig.update_traces(opacity=opacity)
        fig.update_layout(barmode=barmode)
    else:
        fig.update_traces(opacity=1.0)

    norm_label = histnorm if histnorm is not None else "count"
    marginal_label = marginal_mode if marginal_mode is not None else "none"
    title = f"Histogram: {x} ({norm_label}, marginal={marginal_label}"
    if color is not None:
        title += f", barmode={barmode}"
    title += ")"

    fig.update_layout(title=title)
    return fig

def build_count_bar(df: pd.DataFrame, x: str, top_k: int = 20):
    s = df[x].astype("string").fillna("<NA>").value_counts(dropna=False).head(top_k)
    plot_df = s.rename_axis(x).reset_index(name="count")

    fig = px.bar(plot_df, x=x, y="count")
    fig.update_layout(title=f"Count Bar: {x} (top {top_k})", xaxis_tickangle=-45)
    return fig


def build_pareto(df: pd.DataFrame, x: str, top_k: int = 20):
    s = df[x].astype("string").fillna("<NA>").value_counts(dropna=False).head(top_k)
    plot_df = s.rename_axis(x).reset_index(name="count")
    plot_df["cum_pct"] = plot_df["count"].cumsum() / plot_df["count"].sum() * 100

    fig = go.Figure()
    fig.add_bar(x=plot_df[x], y=plot_df["count"], name="Count")
    fig.add_scatter(
        x=plot_df[x],
        y=plot_df["cum_pct"],
        name="Cumulative %",
        yaxis="y2",
        mode="lines+markers",
    )

    fig.update_layout(
        title=f"Pareto: {x} (top {top_k})",
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="Count"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h"),
    )
    return fig


def build_boxplot(df: pd.DataFrame, y: str, x: str | None = None):
    fig = px.box(df, x=x, y=y) if x else px.box(df, y=y)
    fig.update_layout(title=f"Boxplot: {y}" + (f" by {x}" if x else ""))
    return fig


def build_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    facet_col: str | None = None,
    facet_row: str | None = None,
    facet_col_wrap: int | None = None,
    opacity: float = 0.8,
    sample_n: int | None = None,
    color_discrete_map: dict[str, str] | None = None,
):
    plot_df = df.copy()

    if color is not None and color_discrete_map is not None:
        plot_df[color] = plot_df[color].astype("string").fillna("<NA>")

    # Sampling per performance
    if sample_n and len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color,
        size=size,
        facet_col=facet_col,
        facet_row=facet_row,
        facet_col_wrap=facet_col_wrap if (facet_col and not facet_row) else None,
        opacity=opacity,
        color_discrete_map=color_discrete_map,
    )

    fig.update_layout(title=f"Scatter: {x} vs {y}")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def build_corr_heatmap(df: pd.DataFrame, cols: list[str]):
    corr = df[cols].corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(title="Correlation Heatmap")
    return fig

def build_scatter_matrix(
    df: pd.DataFrame,
    cols: list[str],
    color: str | None = None,
    opacity: float = 0.6,
    sample_n: int | None = None,
    diagonal_visible: bool = True,
    show_upper_half: bool = False,
    color_discrete_map: dict[str, str] | None = None,
):
    if len(cols) < 2:
        raise ValueError("Scatter matrix richiede almeno 2 colonne numeriche.")

    plot_df = df.copy()

    if color is not None and color_discrete_map is not None:
        plot_df[color] = plot_df[color].astype("string").fillna("<NA>")

    if sample_n and len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)

    fig = px.scatter_matrix(
        plot_df,
        dimensions=cols,
        color=color,
        opacity=opacity,
        color_discrete_map=color_discrete_map,
    )

    fig.update_traces(
        diagonal_visible=diagonal_visible,
        showupperhalf=show_upper_half,
    )

    fig.update_layout(title=f"Scatter Matrix ({len(cols)} variabili)")
    return fig