# models/train_rf.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from common.utils import get_data, feature_engineering

def train_rf(ticker='BMRI.JK'):
    series = get_data(ticker)
    df = feature_engineering(series)

    split = int(len(df) * 0.85)
    X_train, y_train = df.iloc[:split, 1:], df.iloc[:split, 0]
    X_test, y_test = df.iloc[split:, 1:], df.iloc[split:, 0]

    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing dataset is empty after split")

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    with mlflow.start_run(run_name="RandomForest") as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("MAPE_test", mape)
        mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForest_BMRI")

        print("RandomForest registered. Run ID:", run.info.run_id)

if __name__ == "__main__":
    train_rf()
