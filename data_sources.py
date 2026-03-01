from ISLP import load_data

# Lista iniziale "curata" (poi si espande)
ISLP_DATASETS = [
    "Auto",
    "College",
    "Caravan",
    "Portfolio",
    "Boston",
    "Default",
]

def load_islp_dataset(name: str):
    df = load_data(name)
    return df.copy()
