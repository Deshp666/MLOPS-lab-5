import pandas as pd
import yaml
import mlflow
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from mlflow.models import infer_signature
from logger import setup_logger

logger = setup_logger(__name__)


def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=['selling_price']), df['selling_price']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X.values)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, power_trans


def train_model(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    train_path = config["data"]["train_path"]
    model_path = config["model"]["output_path"]

    df = pd.read_csv(train_path)
    X, y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = config["model"]["params"]

    mlflow.set_experiment("randomforest_model_cars")
    with mlflow.start_run():
        rf = RandomForestRegressor(random_state=42)
        clf = GridSearchCV(rf, params, cv=3, n_jobs=-1)
        clf.fit(X_train, y_train.reshape(-1))

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        best = clf.best_estimator_
        joblib.dump(best, model_path)

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

    logger.info(f"Обученная модель сохранена в {model_path}")


if __name__ == "__main__":
    train_model()
