import pandas as pd
import yaml
from sklearn.preprocessing import OrdinalEncoder
from logger import setup_logger

logger = setup_logger(__name__)


def calculate_outliers(column):
    inter_quantile = column.quantile(0.75) - column.quantile(0.25)
    lower_border = column.quantile(0.25) - inter_quantile * 1.5
    upper_border = column.quantile(0.75) + inter_quantile * 1.5
    return round(lower_border), round(upper_border)


def prepare_dataset(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    raw_path = config["data"]["raw_path"]
    prepared_path = config["data"]["prepared_path"]

    df = pd.read_csv(raw_path)
    logger.info(f"Загружено {df.shape[0]} строк из {raw_path}")

    # Очистка данных
    df = df.drop(['torque'], axis=1)
    df['engine'] = df['engine'].str.replace('CC', '').apply(pd.to_numeric, errors='coerce')
    df['mileage'] = df['mileage'].str.replace('kmpl', '').apply(pd.to_numeric, errors='coerce')
    df['max_power'] = df['max_power'].str.replace('bhp', '').apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Обработка выбросов
    for col in ['year', 'mileage', 'engine', 'max_power', 'km_driven', 'selling_price']:
        lower, upper = calculate_outliers(df[col])
        df.loc[df[col] > upper, col] = upper
        df.loc[df[col] < lower, col] = lower

    # Кодирование категориальных данных
    cat_columns = ['name', 'fuel', 'transmission', 'owner', 'seller_type']
    encoder = OrdinalEncoder()
    df[cat_columns] = encoder.fit_transform(df[cat_columns])

    #Создание новых признаков
    df['distance_by_year'] = round(df['km_driven']/(2021-df['year']))
    df['age'] = 2025 - df['year']

    df.to_csv(prepared_path, index=False)
    logger.info(f"Подготовленный датасет сохранен в {prepared_path}")


if __name__ == "__main__":
    prepare_dataset()
