# registry/select_best_model.py
import mlflow
from mlflow.tracking import MlflowClient

def select_best_model(model_name: str) -> str:
    """
    Memilih model terbaik dari MLflow Model Registry berdasarkan MAPE terendah.
    """
    client = MlflowClient()
    latest_versions = client.search_registered_models(name=model_name)

    best_model_version = None
    best_mape = float('inf')

    for model in latest_versions:
        for version in model.latest_versions:
            if version.status == 'READY':
                # Mengambil metrik MAPE dari model
                mape = version.metrics.get('MAPE_test', float('inf'))
                if mape < best_mape:
                    best_mape = mape
                    best_model_version = version.version

    if best_model_version:
        print(f"Model terbaik: {model_name}, versi {best_model_version} dengan MAPE {best_mape}")
        return f"models:/{model_name}/{best_model_version}"
    else:
        raise ValueError("Tidak ada model yang ditemukan dengan metrik MAPE.")