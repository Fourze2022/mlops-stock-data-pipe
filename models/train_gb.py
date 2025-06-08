import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from common.utils import get_data, feature_engineering

def train_gb(ticker='BMRI.JK'):
    series = get_data(ticker)
    df = feature_engineering(series)

    split = int(len(df) * 0.85)
    X_train, y_train = df.iloc[:split, 1:], df.iloc[:split, 0]
    X_test, y_test = df.iloc[split:, 1:], df.iloc[split:, 0]

    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing dataset is empty after split")

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)

    with mlflow.start_run(run_name="GradientBoosting") as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("MAPE_test", mape)  # seragam dengan select_best_model.py
        mlflow.sklearn.log_model(model, "model", registered_model_name="GradientBoosting_BMRI")

        print("GradientBoosting registered. Run ID:", run.info.run_id)

if __name__ == "__main__":
    train_gb()