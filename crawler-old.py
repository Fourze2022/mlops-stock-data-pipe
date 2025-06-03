import argparse
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import sys
from psycopg2.extras import execute_values

def main(ticker, start_date):
    # 1. Koneksi database (kita pakai SQLAlchemy hanya untuk dapetin raw_connection)
    DB_USER = "stock_user"
    DB_PASS = "stock_pass"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "stockdb"

    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        # test koneksi
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print("Koneksi ke database berhasil.")
    except Exception as e:
        print("Gagal koneksi ke database:", e)
        sys.exit(1)

    # 2. Download data historis via yfinance
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data["ticker"] = ticker
    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(col).strip() for col in data.columns.values]

    # 3. Siapkan DataFrame sesuai schema staging.stock_data_raw
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
    df_staging["ingested_at"] = pd.Timestamp.now()
    df_staging = df_staging[
        ["ticker", "date", "open", "high", "low", "close", "volume", "ingested_at"]
    ]

    # 4. Batch insert manual via psycopg2.extras.execute_values
    row_count = len(df_staging)
    try:c
        # a) Ambil koneksi DBAPI (psycopg2) dari engine
        raw_conn = engine.raw_connection()
        cursor = raw_conn.cursor()

        # b) Pastikan kolom ingested_at jadi datetime Python, bukan pandas.Timestamp
        df_staging["ingested_at"] = df_staging["ingested_at"].apply(lambda ts: ts.to_pydatetime())

        # c) Buat list-of-tuples dengan Python-native types
        tuple_list = list(df_staging.itertuples(index=False, name=None))

        # d) Susun nama kolom dan query INSERT
        cols = df_staging.columns.tolist()
        cols_str = ", ".join(cols)
        insert_sql = f"""
            INSERT INTO staging.stock_data_raw ({cols_str})
            VALUES %s
        """

        # e) Eksekusi batch insert
        from psycopg2.extras import execute_values
        execute_values(cursor, insert_sql, tuple_list)

        raw_conn.commit()
        cursor.close()
        raw_conn.close()

        print(f"Data {ticker} ({start_date} s.d. {end_date}) berhasil di-load ke staging.stock_data_raw.")
        print(f"Jumlah baris yang di-upload: {row_count} baris.")
    except Exception as e:
        print("Gagal menulis ke database (manual batch):", e)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(ticker="BMRI.JK", start_date="2019-01-01")
    else:
        parser = argparse.ArgumentParser(description="Scrape stock data via yfinance.")
        parser.add_argument('--ticker', required=True, help="Kode ticker, misal 'BMRI.JK'")
        parser.add_argument('--start_date', required=True, help="Tanggal mulai (YYYY-MM-DD)")
        args = parser.parse_args()
        main(args.ticker, args.start_date)