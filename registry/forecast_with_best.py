import mlflow.pyfunc
import pandas as pd
from sqlalchemy import create_engine
from common.utils import get_data, feature_engineering

def forecast_and_save(model_uri: str, forecast_days=30):
    # Load model
    model = mlflow.pyfunc.load_model(model_uri)

    # Ambil data historis dan buat fitur
    series = get_data()
    features = feature_engineering(series)

    # Siapkan dataframe untuk prediksi
    last_date = features.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')

    # Buat dataframe untuk menyimpan fitur yang diperbarui
    df_forecast = features.copy()

    preds = []

    for forecast_date in forecast_dates:
        # Ambil fitur terakhir sebagai basis untuk prediksi berikutnya
        last_features = df_forecast.iloc[-1, :].copy()

        # Buat fitur untuk tanggal prediksi baru
        new_features = {}

        # Update lag features: shift lag values, isi lag_1 dengan prediksi terakhir (jika ada)
        for lag in [1, 5, 10]:
            lag_col = f'lag_{lag}'
            if lag == 1:
                # lag_1 adalah prediksi hari sebelumnya (atau terakhir actual)
                new_features[lag_col] = preds[-1] if preds else last_features['Close']
            else:
                # lag_n: ambil dari fitur sebelumnya, geser sesuai lag
                if len(df_forecast) >= lag:
                    new_features[lag_col] = df_forecast.iloc[-lag][
                        'Close' if lag == 0 else f'lag_{lag-1}'
                    ]
                else:
                    new_features[lag_col] = last_features.get(lag_col, last_features['Close'])

        # Update tanggal fitur temporal
        new_features['day_of_year'] = forecast_date.dayofyear
        new_features['month'] = forecast_date.month
        new_features['year'] = forecast_date.year
        new_features['day_of_week'] = forecast_date.dayofweek

        # Buat dataframe fitur baru
        new_features_df = pd.DataFrame(new_features, index=[forecast_date])

        # Prediksi harga penutupan untuk tanggal tersebut
        pred = model.predict(new_features_df)[0]

        # Tambahkan hasil prediksi ke daftar prediksi dan dataframe fitur untuk iterasi berikutnya
        preds.append(pred)

        # Simpan prediksi sebagai 'Close' agar bisa digunakan lag untuk langkah berikutnya
        new_features_df['Close'] = pred

        # Gabungkan ke dataframe fitur untuk iterasi selanjutnya
        df_forecast = pd.concat([df_forecast, new_features_df])

    # Simpan hasil prediksi ke database
    engine = create_engine("postgresql://stock_user:stock_pass@postgres:5432/stockdb")
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'predicted_close': preds
    })

    forecast_df.to_sql('stock_forecasts', engine, if_exists='replace', index=False)
    print("Hasil prediksi berhasil disimpan ke database.")