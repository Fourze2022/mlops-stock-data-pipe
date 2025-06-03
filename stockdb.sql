-- Buat schema
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS production;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Tabel staging.stock_data_raw
CREATE TABLE IF NOT EXISTS staging.stock_data_raw (
  ticker      VARCHAR(10)    NOT NULL,
  date        DATE           NOT NULL,
  open        NUMERIC(12,4),
  high        NUMERIC(12,4),
  low         NUMERIC(12,4),
  close       NUMERIC(12,4),
  volume      BIGINT,
  ingested_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  PRIMARY KEY (ticker, date)
);

-- Tabel production.stock_data_features (nanti dipakai ETL)
CREATE TABLE IF NOT EXISTS production.stock_data_features (
  ticker        VARCHAR(10)    NOT NULL,
  date          DATE           NOT NULL,
  open          NUMERIC(12,4),
  high          NUMERIC(12,4),
  low           NUMERIC(12,4),
  close         NUMERIC(12,4),
  volume        BIGINT,
  ma7           NUMERIC(12,4),
  ma30          NUMERIC(12,4),
  volatility    NUMERIC(12,4),
  roc           NUMERIC(12,4),
  rsi           NUMERIC(5,2),
  processed_at  TIMESTAMP WITH TIME ZONE DEFAULT now(),
  PRIMARY KEY (ticker, date)
);

-- Tabel production.stock_forecasts
CREATE TABLE IF NOT EXISTS production.stock_forecasts (
  ticker        VARCHAR(10)    NOT NULL,
  date          DATE           NOT NULL,
  pred_next1    NUMERIC(12,4),
  pred_next7    NUMERIC(12,4),
  model_name    VARCHAR(50),
  model_version INTEGER,
  predicted_at  TIMESTAMP WITH TIME ZONE DEFAULT now(),
  PRIMARY KEY (ticker, date)
);

-- Tabel monitoring.pipeline_runs
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- untuk gen_random_uuid()
CREATE TABLE IF NOT EXISTS monitoring.pipeline_runs (
  run_id        UUID           PRIMARY KEY DEFAULT gen_random_uuid(),
  run_date      TIMESTAMP WITH TIME ZONE DEFAULT now(),
  status        VARCHAR(20),
  details       TEXT
);

-- Tabel monitoring.model_performance_logs
CREATE TABLE IF NOT EXISTS monitoring.model_performance_logs (
  log_id        UUID           PRIMARY KEY DEFAULT gen_random_uuid(),
  ticker        VARCHAR(10)    NOT NULL,
  model_name    VARCHAR(50),
  model_version INTEGER,
  mape          NUMERIC(5,2),
  rmse          NUMERIC(12,4),
  log_date      TIMESTAMP WITH TIME ZONE DEFAULT now()
);