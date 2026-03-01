import json
from typing import Any


def _py(v: Any) -> str:
    """Rappresentazione Python sicura per valori semplici."""
    if isinstance(v, str):
        return repr(v)
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    return repr(v)


def _imports_block(use_go: bool = False) -> str:
    lines = ["import plotly.express as px"]
    if use_go:
        lines.append("import plotly.graph_objects as go")
    return "\n".join(lines)


def histogram_code(
    x: str,
    nbins: int,
    color: str | None = None,
    opacity: float = 0.75,
    histnorm: str | None = None,
    marginal_mode: str | None = "box",
    barmode: str = "overlay",
    color_discrete_map: dict[str, str] | None = None,
) -> str:
    code = f"""{_imports_block()}
# df = ...  # pandas DataFrame

fig = px.histogram(
    df,
    x={_py(x)},
    nbins={nbins},
    color={_py(color)},
    marginal={_py(marginal_mode)},
    histnorm={_py(histnorm)},
    color_discrete_map={_py(color_discrete_map)},
)
"""
    if color is not None:
        code += f"""fig.update_traces(opacity={opacity})
fig.update_layout(barmode={_py(barmode)})
"""
    else:
        code += """fig.update_traces(opacity=1.0)
"""
    norm_label = histnorm if histnorm is not None else "count"
    marginal_label = marginal_mode if marginal_mode is not None else "none"
    title = f"Histogram: {x} ({norm_label}, marginal={marginal_label}"
    if color is not None:
        title += f", barmode={barmode}"
    title += ")"

    code += f"""fig.update_layout(title={_py(title)})
fig.show()
"""
    return code

def count_bar_code(x: str, top_k: int) -> str:
    return f"""{_imports_block()}
# df = ...  # pandas DataFrame

s = df[{_py(x)}].astype("string").fillna("<NA>").value_counts(dropna=False).head({top_k})
plot_df = s.rename_axis({_py(x)}).reset_index(name="count")

fig = px.bar(plot_df, x={_py(x)}, y="count")
fig.update_layout(title={_py(f"Count Bar: {x} (top {top_k})")}, xaxis_tickangle=-45)
fig.show()
"""


def pareto_code(x: str, top_k: int) -> str:
    return f"""{_imports_block(use_go=True)}
# df = ...  # pandas DataFrame

s = df[{_py(x)}].astype("string").fillna("<NA>").value_counts(dropna=False).head({top_k})
plot_df = s.rename_axis({_py(x)}).reset_index(name="count")
plot_df["cum_pct"] = plot_df["count"].cumsum() / plot_df["count"].sum() * 100

fig = go.Figure()
fig.add_bar(x=plot_df[{_py(x)}], y=plot_df["count"], name="Count")
fig.add_scatter(
    x=plot_df[{_py(x)}],
    y=plot_df["cum_pct"],
    name="Cumulative %",
    yaxis="y2",
    mode="lines+markers"
)

fig.update_layout(
    title={_py(f"Pareto: {x} (top {top_k})")},
    xaxis=dict(tickangle=-45),
    yaxis=dict(title="Count"),
    yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
    legend=dict(orientation="h")
)
fig.show()
"""


def boxplot_code(y: str, x_group: str | None = None) -> str:
    if x_group is None:
        return f"""{_imports_block()}
# df = ...  # pandas DataFrame

fig = px.box(df, y={_py(y)})
fig.update_layout(title={_py(f"Boxplot: {y}")})
fig.show()
"""
    return f"""{_imports_block()}
# df = ...  # pandas DataFrame

fig = px.box(df, x={_py(x_group)}, y={_py(y)})
fig.update_layout(title={_py(f"Boxplot: {y} by {x_group}")})
fig.show()
"""


def scatter_code(
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
) -> str:
    wrap = facet_col_wrap if (facet_col is not None and facet_row is None) else None

    pre = ""
    df_name = "df"
    if sample_n is not None:
        pre = f"""df_plot = df.copy()
if len(df_plot) > {sample_n}:
    df_plot = df_plot.sample({sample_n}, random_state=42)
"""
        df_name = "df_plot"

    code = f"""{_imports_block()}
# df = ...  # pandas DataFrame

{pre}fig = px.scatter(
    {df_name},
    x={_py(x)},
    y={_py(y)},
    color={_py(color)},
    size={_py(size)},
    facet_col={_py(facet_col)},
    facet_row={_py(facet_row)},
    facet_col_wrap={_py(wrap)},
    opacity={opacity},
    color_discrete_map={_py(color_discrete_map)},
)

fig.update_layout(title={_py(f"Scatter: {x} vs {y}")})
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.show()
"""
    return code


def corr_heatmap_code(cols: list[str]) -> str:
    return f"""{_imports_block()}
# df = ...  # pandas DataFrame

cols = {json.dumps(cols, ensure_ascii=False)}
corr = df[cols].corr(numeric_only=True)

fig = px.imshow(corr, text_auto=".2f", aspect="auto")
fig.update_layout(title="Correlation Heatmap")
fig.show()
"""

def scatter_matrix_code(
    cols: list[str],
    color: str | None = None,
    opacity: float = 0.6,
    sample_n: int | None = None,
    diagonal_visible: bool = True,
    show_upper_half: bool = False,
    color_discrete_map: dict[str, str] | None = None,
) -> str:
    cols_repr = repr(cols)

    pre = ""
    df_name = "df"
    if sample_n is not None:
        pre = f"""df_plot = df.copy()
if len(df_plot) > {sample_n}:
    df_plot = df_plot.sample({sample_n}, random_state=42)
"""
        df_name = "df_plot"

    code = f"""{_imports_block()}
# df = ...  # pandas DataFrame

{pre}fig = px.scatter_matrix(
    {df_name},
    dimensions={cols_repr},
    color={_py(color)},
    opacity={opacity},
    color_discrete_map={_py(color_discrete_map)},
)

fig.update_traces(
    diagonal_visible={_py(diagonal_visible)},
    showupperhalf={_py(show_upper_half)},
)

fig.update_layout(title={_py(f"Scatter Matrix ({len(cols)} variabili)")})
fig.show()
"""
    return code