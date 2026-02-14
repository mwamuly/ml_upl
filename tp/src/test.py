import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset.load_data import load_dataset
from src.model.model import get_model
import yaml

def test():
    # Charger config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    df = load_dataset(processed=True)

    X = df.drop("class", axis=1)
    y = df["class"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    # Charger le modèle
    model = joblib.load(config["model_path"])

    y_pred = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
