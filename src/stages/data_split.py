import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def data_split(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    prepared_path = config["data"]["prepared_path"]
    train_path = config["data"]["train_path"]
    test_path = config["data"]["test_path"]
    test_size = config["split"]["test_size"]

    df = pd.read_csv(prepared_path)
    train, test = train_test_split(df, test_size=test_size, random_state=42)

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    logger.info(f"Разделено: train - {train.shape[0]} строк, test - {test.shape[0]} строк")

if __name__ == "__main__":
    data_split()
