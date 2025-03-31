import pandas as pd
import yaml
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from logger import setup_logger

logger = setup_logger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def test_model(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    test_path = config["data"]["test_path"]
    model_path = config["model"]["output_path"]

    df = pd.read_csv(test_path)
    X_test, y_test = df.drop(columns=['selling_price']), df['selling_price']

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    rmse, mae, r2 = eval_metrics(y_test, y_pred)

    logger.info(f"Метрики: RMSE={rmse}, MAE={mae}, R2={r2}")

if __name__ == "__main__":
    test_model()
