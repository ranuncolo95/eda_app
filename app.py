import streamlit as st
import pandas as pd
import plotly.colors as plc

from data_sources import ISLP_DATASETS, load_islp_dataset
from profiling import (
    EDA_TYPES,
    columns_by_eda_type,
    dataset_summary,
    build_column_profile_table,
    apply_eda_types_for_plotting,
    infer_base_eda_types,
)
from plot_builders import (
    build_histogram,
    build_count_bar,
    build_pareto,
    build_boxplot,
    build_scatter,
    build_corr_heatmap,
    build_scatter_matrix,
)

from codegen import (
    histogram_code,
    count_bar_code,
    pareto_code,
    boxplot_code,
    scatter_code,
    corr_heatmap_code,
    scatter_matrix_code,
)

code_snippet = None

st.set_page_config(page_title="EDA ISLP", layout="wide")
st.title("EDA App (Fase 1) — ISLP + pandas + Plotly")

@st.cache_data
def get_dataset(name: str) -> pd.DataFrame:
    return load_islp_dataset(name)

# ---------------------------
# Helpers UI
# ---------------------------
def safe_selectbox(label: str, options: list, key: str | None = None):
    if not options:
        st.warning(f"Nessuna opzione disponibile per: {label}")
        return None
    return st.selectbox(label, options, key=key)

def categorical_color_map_ui(
    df_plot: pd.DataFrame,
    color_col: str | None,
    key_prefix: str,
    max_levels: int = 20,
) -> dict[str, str] | None:
    """
    Restituisce una mappa {categoria: colore_hex} per Plotly oppure None.
    UI: checkbox + palette base + color picker per livello.
    """
    if color_col is None:
        return None

    enable_custom = st.checkbox(
        "Personalizza colori categorie",
        value=False,
        key=f"{key_prefix}_enable_custom_colors",
    )
    if not enable_custom:
        return None

    # Allineato ai builder: string + fillna("<NA>")
    levels = (
        df_plot[color_col]
        .astype("string")
        .fillna("<NA>")
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    levels = sorted(levels)

    if len(levels) == 0:
        st.info("Nessuna categoria disponibile.")
        return None

    if len(levels) > max_levels:
        st.warning(
            f"La variabile '{color_col}' ha {len(levels)} livelli. "
            f"Mostro i primi {max_levels} per evitare una UI troppo lunga."
        )
        levels = levels[:max_levels]

    palette_name = st.selectbox(
        "Palette base",
        ["Plotly", "D3", "G10", "Set2", "Safe"],
        index=0,
        key=f"{key_prefix}_palette_name",
    )
    palette = getattr(plc.qualitative, palette_name)

    st.caption("Scegli il colore per ciascuna categoria:")
    color_map: dict[str, str] = {}

    cols_ui = st.columns(2)
    for i, level in enumerate(levels):
        default_color = palette[i % len(palette)]
        with cols_ui[i % 2]:
            picked = st.color_picker(
                f"{level}",
                value=default_color,
                key=f"{key_prefix}_color_{i}_{level}",
            )
        color_map[level] = picked

    return color_map

def init_state():
    if "type_overrides" not in st.session_state:
        st.session_state["type_overrides"] = {}

init_state()

# ---------------------------
# Dataset selection
# ---------------------------
dataset_name = st.sidebar.selectbox("Dataset ISLP", ISLP_DATASETS)
df = get_dataset(dataset_name)

# Reset override se cambio dataset (opzionale ma consigliato)
if st.session_state.get("_last_dataset") != dataset_name:
    st.session_state["type_overrides"] = {}
    st.session_state["_last_dataset"] = dataset_name

# Tipi base / resolved
groups, resolved_types = columns_by_eda_type(df, st.session_state["type_overrides"])
df_plot, conversion_report = apply_eda_types_for_plotting(df, resolved_types)

# Gruppi "pratici" per i grafici
num_cols = groups["numeric"]
cat_like_cols = groups["categorical"] + groups["string"] + groups["boolean"]
dt_cols = groups["datetime"]

# ---------------------------
# Sidebar: gestione tipi colonne
# ---------------------------
with st.sidebar.expander("Gestione tipi colonne", expanded=False):
    st.caption("Override del tipo EDA per singola colonna (usato da selettori e grafici).")

    col_to_edit = safe_selectbox("Colonna da modificare", df.columns.tolist(), key="type_col_select")

    if col_to_edit is not None:
        base_types = infer_base_eda_types(df)
        current_override = st.session_state["type_overrides"].get(col_to_edit, "auto")

        c1, c2 = st.columns(2)
        c1.write(f"**dtype pandas:** `{df[col_to_edit].dtype}`")
        c2.write(f"**EDA base:** `{base_types[col_to_edit]}`")

        st.write(f"**EDA corrente:** `{resolved_types[col_to_edit]}`")
        st.write("**Anteprima valori:**")
        st.dataframe(df[[col_to_edit]].head(10), use_container_width=True)

        type_options = ["auto"] + EDA_TYPES
        selected_type = st.selectbox(
            "Nuovo tipo EDA",
            type_options,
            index=type_options.index(current_override) if current_override in type_options else 0,
            key="new_eda_type",
        )

        b1, b2, b3 = st.columns(3)
        if b1.button("Applica", use_container_width=True):
            if selected_type == "auto":
                st.session_state["type_overrides"].pop(col_to_edit, None)
            else:
                st.session_state["type_overrides"][col_to_edit] = selected_type
            st.rerun()

        if b2.button("Reset colonna", use_container_width=True):
            st.session_state["type_overrides"].pop(col_to_edit, None)
            st.rerun()

        if b3.button("Reset tutti", use_container_width=True):
            st.session_state["type_overrides"] = {}
            st.rerun()

    if st.session_state["type_overrides"]:
        st.write("**Override attivi**")
        st.json(st.session_state["type_overrides"])
    else:
        st.info("Nessun override attivo.")

# ---------------------------
# Main: overview
# ---------------------------
summary = dataset_summary(df)
m1, m2, m3 = st.columns(3)
m1.metric("Rows", summary["rows"])
m2.metric("Columns", summary["cols"])
m3.metric("Missing cells", summary["missing_cells"])

tab_overview, tab_plot = st.tabs(["Overview", "Plot"])

with tab_overview:
    st.subheader("Preview dati")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Profilo colonne")
    profile_df = build_column_profile_table(df, st.session_state["type_overrides"])
    st.dataframe(profile_df, use_container_width=True)

    with st.expander("Report conversioni per plotting (debug)", expanded=False):
        conv_df = pd.DataFrame(
            [{"column": k, "plot_cast": v} for k, v in conversion_report.items()]
        )
        st.dataframe(conv_df, use_container_width=True)

# ---------------------------
# Main: plot
# ---------------------------
with tab_plot:
    st.subheader("Grafico")

    # Controllo UI per altezza del grafico
    chart_height_user = st.slider(
        "Altezza grafico (px)",
        min_value=500,
        max_value=1600,
        value=800,
        step=50,
        help="Aumenta l'altezza per facet / marginal plot per evitare grafici schiacciati."
    )

    chart_type = st.selectbox(
        "Tipo grafico",
        [
            "Histogram",
            "Count Bar",
            "Pareto",
            "Boxplot",
            "Scatter",
            "Scatter Matrix",
            "Correlation Heatmap",
        ],
    )

    fig = None
    code_snippet = None

    if chart_type == "Histogram":
        if not num_cols:
            st.warning("Nessuna colonna numerica disponibile (in base ai tipi EDA correnti).")
        else:
            x = st.selectbox("Variabile numerica (x)", num_cols)
            color = st.selectbox("Color (opzionale)", [None] + cat_like_cols)
            hist_color_map = categorical_color_map_ui(df_plot, color, key_prefix="hist")
            nbins = st.slider("Bins", 5, 100, 30)

            histnorm_ui = st.selectbox(
                "Normalizzazione",
                ["count", "percent", "probability", "density", "probability density"],
                index=0,
                help="count = frequenze assolute; percent/probability = frequenze relative; density/probability density = densità",
            )
            histnorm = None if histnorm_ui == "count" else histnorm_ui

            marginal_ui = st.selectbox(
                "Marginal mode",
                ["box", "violin", "rug", "none"],
                index=0,
                help="Grafico marginale sopra l'istogramma (per x=...).",
            )
            marginal_mode = None if marginal_ui == "none" else marginal_ui

            opacity = 0.75
            barmode = "overlay"

            if color is not None:
                barmode = st.selectbox(
                    "Bar mode",
                    ["overlay", "group", "relative", "stack"],
                    index=0,
                    help="overlay = sovrapposto; group = affiancato; relative/stack = impilato.",
                )
                opacity = st.slider("Opacità overlay", 0.05, 1.00, 0.55, 0.05)

            fig = build_histogram(
                df_plot,
                x=x,
                nbins=nbins,
                color=color,
                opacity=opacity,
                histnorm=histnorm,
                marginal_mode=marginal_mode,
                barmode=barmode,
                color_discrete_map=hist_color_map,

            )

            code_snippet = histogram_code(
                x=x,
                nbins=nbins,
                color=color,
                opacity=opacity,
                histnorm=histnorm,
                marginal_mode=marginal_mode,
                barmode=barmode,
                color_discrete_map=hist_color_map,

            )

    elif chart_type == "Count Bar":
        if not cat_like_cols:
            st.warning("Nessuna colonna categorica/string/boolean disponibile.")
        else:
            x = st.selectbox("Variabile categorica", cat_like_cols)
            top_k = st.slider("Top K categorie", 5, 100, 20)

            fig = build_count_bar(df_plot, x=x, top_k=top_k)
            code_snippet = count_bar_code(x=x, top_k=top_k)

    elif chart_type == "Pareto":
        if not cat_like_cols:
            st.warning("Nessuna colonna categorica/string/boolean disponibile.")
        else:
            x = st.selectbox("Variabile categorica", cat_like_cols)
            top_k = st.slider("Top K categorie", 5, 100, 20)

            fig = build_pareto(df_plot, x=x, top_k=top_k)
            code_snippet = pareto_code(x=x, top_k=top_k)

    elif chart_type == "Boxplot":
        if not num_cols:
            st.warning("Nessuna colonna numerica disponibile.")
        else:
            y = st.selectbox("Variabile numerica (y)", num_cols)
            x_group = st.selectbox("Grouping (opzionale, cat/string/bool)", [None] + cat_like_cols)

            fig = build_boxplot(df_plot, y=y, x=x_group)
            code_snippet = boxplot_code(y=y, x_group=x_group)

    elif chart_type == "Scatter":
        if len(num_cols) < 2:
            st.warning("Servono almeno 2 colonne numeriche.")
        else:
            x = st.selectbox("X (numerica)", num_cols)
            y_candidates = [c for c in num_cols if c != x]
            y = st.selectbox("Y (numerica)", y_candidates)

            facet_candidates = [None] + cat_like_cols
            color = st.selectbox("Color (opzionale)", facet_candidates)
            scatter_color_map = categorical_color_map_ui(df_plot, color, key_prefix="scatter")
            facet_col = st.selectbox("Facet col (opzionale)", facet_candidates)
            facet_row = st.selectbox("Facet row (opzionale)", facet_candidates)

            if facet_row is not None and facet_col is not None and facet_row == facet_col:
                st.warning("Facet row e facet col non dovrebbero essere la stessa colonna.")

            opacity = st.slider("Opacità punti", 0.05, 1.00, 0.30, 0.05)

            facet_col_wrap = None
            if facet_col is not None and facet_row is None:
                use_wrap = st.checkbox("Usa facet_col_wrap", value=False)
                if use_wrap:
                    facet_col_wrap = st.slider("Facet col wrap", 2, 6, 3)

            sample_n = st.number_input("Sample max (0 = no sample)", min_value=0, value=3000, step=100)
            sample_n = None if sample_n == 0 else int(sample_n)

            too_many = []
            for fc_name, fc_val in [("facet_col", facet_col), ("facet_row", facet_row)]:
                if fc_val is not None:
                    n_levels = int(df_plot[fc_val].astype("string").nunique(dropna=True))
                    if n_levels > 12:
                        too_many.append((fc_name, fc_val, n_levels))

            if too_many:
                msg = " | ".join([f"{n}='{c}' ha {k} livelli" for n, c, k in too_many])
                st.warning(f"Attenzione: troppi facet possono rendere il grafico poco leggibile ({msg}).")

            fig = build_scatter(
                df_plot,
                x=x,
                y=y,
                color=color,
                facet_col=facet_col,
                facet_row=facet_row,
                facet_col_wrap=facet_col_wrap,
                opacity=opacity,
                sample_n=sample_n,
                color_discrete_map=scatter_color_map,

            )
            code_snippet = scatter_code(
                x=x,
                y=y,
                color=color,
                facet_col=facet_col,
                facet_row=facet_row,
                facet_col_wrap=facet_col_wrap,
                opacity=opacity,
                sample_n=sample_n,
                color_discrete_map=scatter_color_map,

            )

    elif chart_type == "Scatter Matrix":
        if len(num_cols) < 2:
            st.warning("Servono almeno 2 colonne numeriche.")
        else:
            default_cols = num_cols[: min(5, len(num_cols))]
            cols = st.multiselect(
                "Colonne numeriche (2-8 consigliate)",
                num_cols,
                default=default_cols,
                help="Troppe colonne rendono la scatter matrix lenta e poco leggibile.",
            )

            color = st.selectbox("Color (opzionale)", [None] + cat_like_cols)
            matrix_color_map = categorical_color_map_ui(df_plot, color, key_prefix="scatter_matrix")

            opacity = st.slider("Opacità punti (matrix)", 0.05, 1.00, 0.35, 0.05)

            diagonal_visible = st.checkbox("Mostra diagonale", value=True)
            show_upper_half = st.checkbox("Mostra triangolo superiore", value=False)

            sample_n = st.number_input(
                "Sample max (0 = no sample)",
                min_value=0,
                value=2000,
                step=100,
                key="scatter_matrix_sample_n",
            )
            sample_n = None if sample_n == 0 else int(sample_n)

            if len(cols) < 2:
                st.warning("Seleziona almeno 2 colonne numeriche.")
            elif len(cols) > 8:
                st.warning("Hai selezionato molte variabili: il grafico può diventare lento o difficile da leggere.")
                fig = build_scatter_matrix(
                    df_plot,
                    cols=cols,
                    color=color,
                    opacity=opacity,
                    sample_n=sample_n,
                    diagonal_visible=diagonal_visible,
                    show_upper_half=show_upper_half,
                    color_discrete_map=matrix_color_map,

                )
                code_snippet = scatter_matrix_code(
                    cols=cols,
                    color=color,
                    opacity=opacity,
                    sample_n=sample_n,
                    diagonal_visible=diagonal_visible,
                    show_upper_half=show_upper_half,
                    color_discrete_map=matrix_color_map,

                )
            else:
                fig = build_scatter_matrix(
                    df_plot,
                    cols=cols,
                    color=color,
                    opacity=opacity,
                    sample_n=sample_n,
                    diagonal_visible=diagonal_visible,
                    show_upper_half=show_upper_half,
                    color_discrete_map=matrix_color_map,

                )
                code_snippet = scatter_matrix_code(
                    cols=cols,
                    color=color,
                    opacity=opacity,
                    sample_n=sample_n,
                    diagonal_visible=diagonal_visible,
                    show_upper_half=show_upper_half,
                    color_discrete_map=matrix_color_map,

                )
                
    elif chart_type == "Correlation Heatmap":
        if len(num_cols) < 2:
            st.warning("Servono almeno 2 colonne numeriche.")
        else:
            default_cols = num_cols[: min(8, len(num_cols))]
            cols = st.multiselect("Colonne numeriche", num_cols, default=default_cols)

            if len(cols) >= 2:
                fig = build_corr_heatmap(df_plot, cols)
                code_snippet = corr_heatmap_code(cols=cols)
            else:
                st.warning("Seleziona almeno 2 colonne numeriche per la heatmap.")

    # -----------------------
    # Render output + code snippet
    # -----------------------
    if fig is not None:
        fig.update_layout(
            height=chart_height_user,
            margin=dict(l=30, r=30, t=70, b=30),
        )

        st.plotly_chart(fig, width="stretch")
        st.markdown("### Codice Python (copiabile nel notebook)")
        st.code(code_snippet or "# Nessun codice disponibile", language="python")