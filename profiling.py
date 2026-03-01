import pandas as pd

EDA_TYPES = ["numeric", "categorical", "boolean", "datetime", "string"]


def infer_base_eda_types(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for col in df.columns:
        s = df[col]

        if pd.api.types.is_bool_dtype(s):
            out[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = "numeric"
        elif pd.api.types.is_categorical_dtype(s) or s.dtype == "object" or pd.api.types.is_string_dtype(s):
            out[col] = "categorical"
        else:
            out[col] = "string"
    return out


def resolve_eda_types(df: pd.DataFrame, overrides: dict[str, str] | None = None) -> dict[str, str]:
    resolved = infer_base_eda_types(df)
    overrides = overrides or {}

    for col, t in overrides.items():
        if col in resolved and t in EDA_TYPES:
            resolved[col] = t
    return resolved


def columns_by_eda_type(
    df: pd.DataFrame, overrides: dict[str, str] | None = None
) -> tuple[dict[str, list[str]], dict[str, str]]:
    resolved = resolve_eda_types(df, overrides)
    groups: dict[str, list[str]] = {k: [] for k in EDA_TYPES}

    for col, t in resolved.items():
        groups.setdefault(t, []).append(col)

    return groups, resolved


def apply_eda_types_for_plotting(
    df: pd.DataFrame, resolved_types: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Applica conversioni leggere per allineare il comportamento dei plot ai tipi EDA.
    Restituisce:
      - dataframe "plot-ready"
      - report conversioni (ok / warning)
    """
    out = df.copy()
    report: dict[str, str] = {}

    for col, eda_t in resolved_types.items():
        if col not in out.columns:
            continue

        try:
            if eda_t == "numeric":
                before_notna = out[col].notna().sum()
                out[col] = pd.to_numeric(out[col], errors="coerce")
                after_notna = out[col].notna().sum()
                lost = int(before_notna - after_notna)
                report[col] = f"numeric (coerce: {lost})" if lost > 0 else "numeric"
            elif eda_t == "datetime":
                before_notna = out[col].notna().sum()
                out[col] = pd.to_datetime(out[col], errors="coerce")
                after_notna = out[col].notna().sum()
                lost = int(before_notna - after_notna)
                report[col] = f"datetime (coerce: {lost})" if lost > 0 else "datetime"
            elif eda_t in {"categorical", "string", "boolean"}:
                # Cast a string per forzare comportamento discreto nei plot
                out[col] = out[col].astype("string")
                report[col] = eda_t
            else:
                report[col] = "unchanged"
        except Exception as e:
            report[col] = f"warning: {e}"

    return out, report


def dataset_summary(df: pd.DataFrame):
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
    }


def build_column_profile_table(
    df: pd.DataFrame,
    overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    overrides = overrides or {}
    base = infer_base_eda_types(df)
    resolved = resolve_eda_types(df, overrides)

    rows = []
    for col in df.columns:
        s = df[col]
        n = len(s)
        missing = int(s.isna().sum())
        missing_pct = (missing / n * 100) if n else 0.0
        try:
            nunique = int(s.nunique(dropna=True))
        except Exception:
            nunique = None

        rows.append(
            {
                "column": col,
                "pandas_dtype": str(s.dtype),
                "base_eda_type": base[col],
                "override": overrides.get(col, "auto"),
                "eda_type": resolved[col],
                "missing_pct": round(missing_pct, 2),
                "nunique": nunique,
            }
        )

    return pd.DataFrame(rows)