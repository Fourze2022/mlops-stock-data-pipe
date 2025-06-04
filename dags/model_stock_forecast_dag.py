# File: dags/model_stock_forecast_dag.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='model_stock_forecast',
    default_args=default_args,
    description='DAG untuk training & forecasting model BMRI setelah extract selesai',
    schedule_interval='@daily',       # jalan sekali sehari, sama seperti DAG Extract
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['model', 'forecast']
) as dag:

    # 1. Task pertama: tunggu DAG extract_stock_data selesai "run_crawler_script"
    wait_for_extract = ExternalTaskSensor(
        task_id='wait_for_extract',
        external_dag_id='extract_stock_data',       # DAG yang ditunggu
        external_task_id='run_crawler_script',      # task di DAG Extract
        start_date=datetime(2024, 1, 1),
        poke_interval=60,      # cek setiap 60 detik
        timeout=60 * 60 * 2,   # maksimal menunggu 2 jam (sesuaikan kalau butuh lebih lama)
        mode='poke'            # mode poke (bisa juga 'reschedule', tapi poke lebih sederhana)
    )

    # 2. Task kedua: setelah extract selesai, jalankan model.py
    run_model = BashOperator(
        task_id='run_model_script',
        bash_command='python /opt/airflow/scripts/model.py'
    )

    # Atur dependency: tunggu dulu extract, baru run model
    wait_for_extract >> run_model