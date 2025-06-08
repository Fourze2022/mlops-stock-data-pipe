import pandas as pd
from sqlalchemy import create_engine

def get_data(ticker='BMRI.JK'):
    engine = create_engine("postgresql://stock_user:stock_pass@postgres:5432/stockdb")
    query = f"""
        SELECT date, close
        FROM staging.stock_data_raw
        WHERE ticker = '{ticker}'
        ORDER BY date
    """
    df = pd.read_sql(query, engine, parse_dates=["date"])
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    df.set_index("date", inplace=True)
    return df["close"]

def feature_engineering(series):
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        series.index = pd.to_datetime(series.index)
    df = series.to_frame(name="Close")
    for lag in [1, 5, 10]:
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["day_of_week"] = df.index.dayofweek
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Feature engineering resulted in empty dataframe")
    return df