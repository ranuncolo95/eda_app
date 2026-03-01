import os
import re
import json
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from ISLP import load_data

# Lista iniziale "curata"
ISLP_DATASETS = [
    "Auto", "Bikeshare", "Boston", "BrainCancer", "Caravan",
    "Carseats", "College", "Credit", "Default", "Fund",
    "Hitters", "Khan", "NCI60", "NYSE", "OJ",
    "Portfolio", "Publication", "Smarket", "Wage", "Weekly",
]


def load_islp_dataset(name: str) -> pd.DataFrame:
    df = load_data(name)
    return df.copy()


def load_csv_uploaded(file_obj, **read_csv_kwargs) -> pd.DataFrame:
    return pd.read_csv(file_obj, **read_csv_kwargs)


def parse_kaggle_dataset_slug(url_or_slug: str) -> str:
    """
    Accetta:
      - slug diretto: 'owner/dataset-name'
      - URL kaggle dataset: https://www.kaggle.com/datasets/owner/dataset-name
    Restituisce: 'owner/dataset-name'
    """
    s = (url_or_slug or "").strip()

    if re.fullmatch(r"[^/\s]+/[^/\s]+", s):
        return s

    m = re.search(r"kaggle\.com/datasets/([^/\s]+)/([^/?#\s]+)", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    raise ValueError(
        "Formato non valido. Inserisci uno slug 'owner/dataset' oppure un link Kaggle tipo "
        "'https://www.kaggle.com/datasets/owner/dataset'."
    )


def _read_kaggle_json_credentials(kaggle_json_path: str | Path) -> tuple[str, str]:
    p = Path(kaggle_json_path).expanduser().resolve()

    if not p.exists():
        raise FileNotFoundError(f"kaggle.json non trovato: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Impossibile leggere/parsing kaggle.json: {e}") from e

    username = (data.get("username") or "").strip()
    key = (data.get("key") or "").strip()

    if not username or not key:
        raise ValueError("kaggle.json deve contenere chiavi 'username' e 'key' non vuote.")

    return username, key


def _kaggle_api_from_json_path(kaggle_json_path: str | Path):
    """
    Crea un client Kaggle API leggendo credenziali da kaggle.json locale.
    """
    username, key = _read_kaggle_json_credentials(kaggle_json_path)

    # Variabili d'ambiente per la libreria kaggle
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError(
            "Package 'kaggle' non disponibile. Installa con: pip install kaggle"
        ) from e

    api = KaggleApi()
    api.authenticate()
    return api


def download_kaggle_dataset_to_tempdir(
    dataset_url_or_slug: str,
    kaggle_json_path: str | Path = "kaggle.json",
) -> tuple[str, str]:
    """
    Scarica e unzip un dataset Kaggle in una cartella temporanea.
    Restituisce: (dataset_slug, temp_dir_path)
    """
    slug = parse_kaggle_dataset_slug(dataset_url_or_slug)
    api = _kaggle_api_from_json_path(kaggle_json_path)

    tmpdir = tempfile.mkdtemp(prefix="eda_kaggle_")
    api.dataset_download_files(slug, path=tmpdir, unzip=True, quiet=True)

    return slug, tmpdir


def list_csv_files_in_dir(dir_path: str) -> list[str]:
    p = Path(dir_path)
    return sorted([str(x) for x in p.rglob("*.csv")])


def load_csv_from_path(path: str, **read_csv_kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **read_csv_kwargs)