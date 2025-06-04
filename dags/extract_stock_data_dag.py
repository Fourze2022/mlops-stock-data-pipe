# File: dags/extract_stock_data_dag.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='extract_stock_data',
    default_args=default_args,
    description='DAG untuk scraping data saham BMRI via yfinance',
    schedule_interval='@daily',       # satu kali setiap hari (00:00)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['extract']
) as dag:

    run_crawler = BashOperator(
        task_id='run_crawler_script',
        bash_command='python /opt/airflow/scripts/crawler.py --ticker BMRI.JK'
    )

    run_crawler