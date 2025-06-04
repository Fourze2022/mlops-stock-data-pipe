import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import mlflow
import warnings
import sys

warnings.filterwarnings("ignore")

def main():
    # --------------------------------------------
    # 1. Inisialisasi koneksi database
    # --------------------------------------------
    DB_USER = "stock_user"
    DB_PASS = "stock_pass"
    DB_HOST = "postgres"
    DB_PORT = "5432"
    DB_NAME = "stockdb"

    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Koneksi ke database berhasil.")
    except Exception as e:
        print("Gagal koneksi ke database:", e)
        sys.exit(1)

    # --------------------------------------------
    # 2. Load data mentah langsung dari staging.stock_data_raw
    # --------------------------------------------
    query = """
        SELECT date, close
        FROM staging.stock_data_raw
        WHERE ticker = 'BMRI.JK'
        ORDER BY date
    """
    df_raw = pd.read_sql(query, engine, parse_dates=["date"])
    if df_raw.empty:
        print("Tidak ada data di staging.stock_data_raw untuk ticker BMRI.JK.")
        sys.exit(1)

    df_raw.set_index("date", inplace=True)
    series = df_raw["close"].copy()

    # --------------------------------------------
    # 3. Feature Engineering (lag dan time features)
    # --------------------------------------------
    ml_data = series.to_frame(name="Close_BMRI.JK").copy()
    for lag in [1, 5, 10]:
        ml_data[f"lag_{lag}"] = ml_data["Close_BMRI.JK"].shift(lag)

    ml_data["day_of_year"] = ml_data.index.dayofyear
    ml_data["month"] = ml_data.index.month
    ml_data["year"] = ml_data.index.year
    ml_data["day_of_week"] = ml_data.index.dayofweek
    ml_data.dropna(inplace=True)

    # --------------------------------------------
    # 4. Train-Test Split (85% train, 15% test)
    # --------------------------------------------
    split_idx = int(len(ml_data) * 0.85)
    train_ml = ml_data.iloc[:split_idx]
    test_ml = ml_data.iloc[split_idx:]

    X_train_ml = train_ml.drop(columns=["Close_BMRI.JK"])
    y_train_ml = train_ml["Close_BMRI.JK"]
    X_test_ml = test_ml.drop(columns=["Close_BMRI.JK"])
    y_test_ml = test_ml["Close_BMRI.JK"]

    # --------------------------------------------
    # 5. Inisialisasi model
    # --------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=10
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42, learning_rate=0.05, max_depth=5
        )
    }

    # --------------------------------------------
    # 6. Tentukan future dates (30 business days setelah data terakhir)
    # --------------------------------------------
    last_date = ml_data.index[-1]
    future_steps_ml = 30
    future_dates_ml = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_steps_ml,
        freq="B"
    )

    # --------------------------------------------
    # 7. Persiapkan DataFrame hasil (historis + future)
    # --------------------------------------------
    full_index = series.index.union(future_dates_ml)
    results_df = pd.DataFrame(index=full_index)
    results_df["Actual"] = series  # hanya historis yang ada actual
    error_metrics = {}

    # --------------------------------------------
    # 8. Training, Forecasting, Logging ke MLflow, dan Simpan ke DB
    # --------------------------------------------
    for model_name, model in models.items():
        print(f"\nTraining dan forecasting model: {model_name}")

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            # Log parameter ke MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_test_split_ratio", f"{split_idx}/{len(ml_data)}")
            mlflow.log_param("lag_features", [1, 5, 10])
            if model_name == "Random Forest":
                mlflow.log_param("n_estimators", model.get_params()["n_estimators"])
                mlflow.log_param("max_depth", model.get_params()["max_depth"])
                mlflow.log_param("min_samples_split", model.get_params()["min_samples_split"])
            elif model_name == "Gradient Boosting":
                mlflow.log_param("n_estimators", model.get_params()["n_estimators"])
                mlflow.log_param("learning_rate", model.get_params()["learning_rate"])
                mlflow.log_param("max_depth", model.get_params()["max_depth"])

            # --- A. Fit model ---
            model.fit(X_train_ml, y_train_ml)

            # --- B. Predict untuk training & test set ---
            train_pred_ml = model.predict(X_train_ml)
            test_pred_ml = model.predict(X_test_ml)
            results_df.loc[y_train_ml.index, f"{model_name}_Train_Forecast"] = train_pred_ml
            results_df.loc[y_test_ml.index, f"{model_name}_Test_Forecast"] = test_pred_ml

            # --- C. Siapkan fitur future (30 hari ke depan) ---
            future_df = pd.DataFrame(index=future_dates_ml)
            temp_concat = pd.concat([ml_data, pd.DataFrame(index=future_dates_ml)])

            for lag in [1, 5, 10]:
                future_df[f"lag_{lag}"] = temp_concat["Close_BMRI.JK"].shift(lag).loc[future_dates_ml]
                if future_df[f"lag_{lag}"].isnull().any():
                    last_val = series.iloc[-lag] if len(series) >= lag else series.iloc[-1]
                    future_df[f"lag_{lag}"].fillna(last_val, inplace=True)

            future_df["day_of_year"] = future_df.index.dayofyear
            future_df["month"] = future_df.index.month
            future_df["year"] = future_df.index.year
            future_df["day_of_week"] = future_df.index.dayofweek
            future_df = future_df[X_train_ml.columns]

            # --- D. Forecast 30 hari ke depan ---
            future_pred_ml = model.predict(future_df)
            results_df.loc[future_dates_ml, f"{model_name}_Forecast"] = future_pred_ml

            # --- E. Hitung metrik error pada test set ---
            mae_ml = mean_absolute_error(y_test_ml, test_pred_ml)
            mse_ml = mean_squared_error(y_test_ml, test_pred_ml)
            rmse_ml = np.sqrt(mse_ml)
            mape_ml = mean_absolute_percentage_error(y_test_ml, test_pred_ml)
            error_metrics[model_name] = {
                "MAE": mae_ml, "MSE": mse_ml, "RMSE": rmse_ml, "MAPE": mape_ml
            }

            # Log metrik ke MLflow
            mlflow.log_metric("MAE_test", mae_ml)
            mlflow.log_metric("RMSE_test", rmse_ml)
            mlflow.log_metric("MAPE_test", mape_ml)

            # --------------------------------------------
            # F. Simpan hasil future forecast ke production.stock_forecasts (batch insert)
            # --------------------------------------------
            try:
                raw_conn = engine.raw_connection()
                cursor = raw_conn.cursor()

                df_to_db = pd.DataFrame({
                    "ticker": ["BMRI.JK"] * len(future_dates_ml),
                    "date": future_dates_ml.date,
                    "model_name": [model_name] * len(future_dates_ml),
                    "model_version": [1] * len(future_dates_ml),
                    "predicted": future_pred_ml,
                    "predicted_at": [pd.Timestamp.now().to_pydatetime()] * len(future_dates_ml),
                    "mlflow_run_id": [run_id] * len(future_dates_ml)
                })

                tuples = list(df_to_db.itertuples(index=False, name=None))
                cols = ",".join(df_to_db.columns.tolist())
                insert_sql = f"""
                    INSERT INTO production.stock_forecasts ({cols})
                    VALUES %s
                    ON CONFLICT (ticker, date, model_name) DO UPDATE 
                      SET predicted = EXCLUDED.predicted,
                          predicted_at = EXCLUDED.predicted_at,
                          mlflow_run_id = EXCLUDED.mlflow_run_id
                """
                execute_values(cursor, insert_sql, tuples)
                raw_conn.commit()
                cursor.close()
                raw_conn.close()
                print(f"{model_name}: Forecast 30 hari berhasil disimpan ke production.stock_forecasts.")
            except Exception as e:
                print(f"Gagal menyimpan forecast ke DB untuk {model_name}:", e)

            # --------------------------------------------
            # G. Simpan metrik ke monitoring.model_performance_logs
            # --------------------------------------------
            try:
                raw_conn2 = engine.raw_connection()
                cursor2 = raw_conn2.cursor()

                df_perf = pd.DataFrame({
                    "ticker": ["BMRI.JK"],
                    "model_name": [model_name],
                    "model_version": [1],
                    "mape": [mape_ml],
                    "rmse": [rmse_ml],
                    "mlflow_run_id": [run_id],
                    "log_date": [pd.Timestamp.now().to_pydatetime()]
                })

                tuples_perf = list(df_perf.itertuples(index=False, name=None))
                cols_perf = ",".join(df_perf.columns.tolist())
                insert_perf_sql = f"""
                    INSERT INTO monitoring.model_performance_logs ({cols_perf})
                    VALUES %s
                """
                execute_values(cursor2, insert_perf_sql, tuples_perf)
                raw_conn2.commit()
                cursor2.close()
                raw_conn2.close()
                print(f"{model_name}: Metrik disimpan ke monitoring.model_performance_logs.")
            except Exception as e:
                print(f"Gagal menyimpan metrik performa untuk {model_name}:", e)


if __name__ == "__main__":
    main()