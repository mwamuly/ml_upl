import pandas as pd
import os
import yaml
from sklearn.preprocessing import LabelEncoder

def load_dataset(processed=True):
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    if processed and os.path.exists(config["processed_data_path"]):
        # Chargement des données traitées
        df = pd.read_csv(config["processed_data_path"])
    else:
        # Chargement des données b
        columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
        df = pd.read_csv(config["raw_data_path"], header=None, names=columns)

        #la nous Encodons les classe
        le = LabelEncoder()
        df["class"] = le.fit_transform(df["class"])

        #---''''' Sauvegarder dans processed
        os.makedirs(os.path.dirname(config["processed_data_path"]), exist_ok=True)
        df.to_csv(config["processed_data_path"], index=False)

    return df
