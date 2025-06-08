import streamlit as st
import pandas as pd
from common.utils import get_data
from sqlalchemy import create_engine
from registry.select_best_model import select_best_model
import mlflow

TICKER = "BMRI.JK"

def get_engine():
    return create_engine("postgresql://stock_user:stock_pass@postgres:5432/stockdb")

def load_forecast_data():
    engine = get_engine()
    query = "SELECT * FROM stock_forecasts ORDER BY date"
    return pd.read_sql(query, engine, parse_dates=["date"])

def merge_historical_forecast(historical, forecast):
    df_hist = historical.rename("close").to_frame()
    df_forecast = forecast.set_index("date")["predicted_close"].rename("forecast")
    combined = pd.concat([df_hist, df_forecast], axis=0)
    return combined

def get_best_model_info(model_name):
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models(name=model_name)
    best_version = None
    best_mape = float("inf")
    for model in models:
        for v in model.latest_versions:
            if v.status == "READY":
                mape = v.metrics.get("MAPE_test", float("inf"))
                if mape < best_mape:
                    best_mape = mape
                    best_version = v.version
    if best_version is None:
        return None, None, None
    model_uri = f"models:/{model_name}/{best_version}"
    return best_version, best_mape, model_uri

st.set_page_config(page_title="Dashboard Saham BMRI", layout="wide")

st.title("Dashboard Harga Saham BMRI")

try:
    historical_data = get_data(TICKER)
    forecast_df = load_forecast_data()
    combined = merge_historical_forecast(historical_data, forecast_df)

    # Layout 2 kolom untuk grafik & detail model
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Harga Penutupan Historis dan Prediksi 30 Hari ke Depan")
        st.line_chart(combined)

    with col2:
        st.subheader("Info Model Terbaik")
        model_name = "GradientBoosting_BMRI"
        version, mape, uri = get_best_model_info(model_name)
        if version:
            st.markdown(f"**Model:** {model_name}")
            st.markdown(f"**Versi:** {version}")
            st.markdown(f"**MAPE:** {mape:.4f}")
            st.markdown(f"Model URI: `{uri}`")
        else:
            st.warning("Model terbaik belum ditemukan.")

    st.markdown("---")
    st.subheader("Tabel Hasil Prediksi")
    st.dataframe(forecast_df, height=300)

except Exception as e:
    st.error(f"Gagal memuat data: {e}")
