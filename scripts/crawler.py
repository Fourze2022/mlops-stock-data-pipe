import argparse
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
import sys
from psycopg2.extras import execute_values

def main(ticker):
    # 1. Koneksi database (kita pakai SQLAlchemy hanya untuk dapetin raw_connection)
    DB_USER = "stock_user"
    DB_PASS = "stock_pass"
    DB_HOST = "postgres"
    DB_PORT = "5432"
    DB_NAME = "stockdb"

    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        # test koneksi
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Koneksi ke database berhasil.")
    except Exception as e:
        print("Gagal koneksi ke database:", e)
        sys.exit(1)

    # 2. Hitung tanggal hari ini (YYYY-MM-DD) untuk start_date dan end_date
    today = pd.Timestamp.now().normalize()  # midnight hari ini
    start_date = today.strftime("%Y-%m-%d")
    end_date = start_date  # jika hanya ingin data hari ini

    # 3. Download data historis via yfinance (hanya tanggal hari ini)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        print(f"Tidak ada data untuk {ticker} pada tanggal {start_date}.")
        return

    data["ticker"] = ticker
    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(col).strip() for col in data.columns.values]

    # 4. Siapkan DataFrame sesuai schema staging.stock_data_raw
    try:
        df_staging = data[[
            "Date_",
            f"Open_{ticker}",
            f"High_{ticker}",
            f"Low_{ticker}",
            f"Close_{ticker}",
            f"Volume_{ticker}"
        ]].copy()
    except KeyError as ke:
        print("Kolom tidak ditemukan di DataFrame:", ke)
        sys.exit(1)

    df_staging.columns = [
        "date", "open", "high", "low", "close", "volume"
    ]
    df_staging["ticker"] = ticker
    df_staging["date"] = pd.to_datetime(df_staging["date"]).dt.date
    df_staging["ingested_at"] = pd.Timestamp.now().to_pydatetime()
    df_staging = df_staging[
        ["ticker", "date", "open", "high", "low", "close", "volume", "ingested_at"]
    ]

    # 5. Batch insert manual via psycopg2.extras.execute_values
    row_count = len(df_staging)
    try:
        raw_conn = engine.raw_connection()
        cursor = raw_conn.cursor()

        # Pastikan semua nilai sudah jadi Python-native (datetime, int, float)
        # (kolom ingested_at sudah di-convert di atas)

        # Buat list-of-tuples
        tuple_list = list(df_staging.itertuples(index=False, name=None))

        # Susun nama kolom dan query INSERT
        cols = df_staging.columns.tolist()
        cols_str = ", ".join(cols)
        insert_sql = f"""
            INSERT INTO staging.stock_data_raw ({cols_str})
            VALUES %s
        """

        execute_values(cursor, insert_sql, tuple_list)
        raw_conn.commit()
        cursor.close()
        raw_conn.close()

        print(f"Data {ticker} ({start_date}) berhasil di-load ke staging.stock_data_raw.")
        print(f"Jumlah baris yang di-upload: {row_count} baris.")
    except Exception as e:
        print("Gagal menulis ke database (manual batch):", e)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape data saham harian via yfinance (mulai dari tanggal hari ini)."
    )
    parser.add_argument(
        '--ticker',
        required=True,
        help="Kode ticker, misal 'BMRI.JK'"
    )
    args = parser.parse_args()
    main(args.ticker)