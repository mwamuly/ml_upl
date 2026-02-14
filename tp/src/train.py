import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.dataset.load_data import load_dataset
from src.model.model import get_model
import yaml

def train():

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Charger les données traitées
    df = load_dataset(processed=True)

    X = df.drop("class", axis=1)
    y = df["class"]

    # Normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    # Créer et entraîner le modèle
    model = get_model()
    model.fit(X_train, y_train)

    # Évaluer
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Training Accuracy:", acc)

    # Sauvegarder le modèle
    joblib.dump(model, config["model_path"])
    print("Model saved at", config["model_path"])

    return model
