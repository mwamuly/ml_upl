from sklearn.ensemble import RandomForestClassifier
import yaml

def get_model():
   
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    return RandomForestClassifier(
        n_estimators=config["n_estimators"],
        random_state=config["random_state"]
    )
