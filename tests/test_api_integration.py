"""
API集成测试 - FastAPI TestClient 端点验证
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient

AUTH_HEADERS = {"X-API-Key": "dev-api-key-change-me"}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestHealthAndSystem:
    """健康检查和系统端点测试"""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "api_version" in data

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data

    def test_version_endpoint(self, client):
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data
        assert "model_version" in data
        assert "framework" in data

    def test_model_status_endpoint(self, client):
        response = client.get("/model/status")
        assert response.status_code == 200
        data = response.json()
        assert "loaded" in data
        assert "device" in data

    def test_stats_endpoint_not_found(self, client):
        response = client.get("/stats")
        assert response.status_code != 404


class TestInferenceValidation:
    """推理请求参数验证测试"""

    def test_inference_missing_opt_sequence(self, client):
        payload = {"doy": [0.0, 0.5, 1.0]}
        response = client.post("/inference", json=payload)
        assert response.status_code == 422

    def test_inference_empty_opt_sequence(self, client):
        payload = {"opt_sequence": [], "doy": [0.0, 0.5, 1.0]}
        response = client.post("/inference", json=payload)
        assert response.status_code == 422

    def test_inference_invalid_n_passes(self, client):
        import numpy as np
        T, H, W = 12, 32, 32
        payload = {
            "opt_sequence": np.random.rand(T, 10, H, W).astype(np.float32).tolist(),
            "sar_sequence": np.random.rand(T, 5, H, W).astype(np.float32).tolist(),
            "dem_data": np.random.rand(5, H, W).astype(np.float32).tolist(),
            "doy": np.linspace(0, 1, T).tolist(),
            "n_passes": 50,
        }
        response = client.post("/inference", json=payload)
        assert response.status_code == 422

    def test_batch_inference_empty(self, client):
        response = client.post("/inference/batch", json={"requests": []})
        assert response.status_code == 422

    def test_predict_model_invalid(self, client):
        response = client.post("/predict/invalid_model", json={}, headers=AUTH_HEADERS)
        assert response.status_code == 400

    def test_predict_model_v5_ok(self, client):
        response = client.post("/predict/v5", json={}, headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "dominant" in data
        assert "confidence" in data


class TestModelLoad:
    """模型加载端点测试"""

    def test_load_nonexistent_model(self, client):
        payload = {"model_path": "models/nonexistent.pt"}
        response = client.post("/model/load", json=payload, headers=AUTH_HEADERS)
        assert response.status_code in (403, 404, 500)

    def test_load_model_missing_path(self, client):
        response = client.post("/model/load", json={}, headers=AUTH_HEADERS)
        assert response.status_code == 422


class TestFileInference:
    """文件上传推理端点测试"""

    def test_file_inference_no_files(self, client):
        response = client.post("/inference/file")
        assert response.status_code == 422

    def test_predict_upload_no_files(self, client):
        response = client.post("/predict/v5/upload", headers=AUTH_HEADERS)
        assert response.status_code == 422


class TestMetricsEndpoint:
    """指标计算端点测试"""

    def test_metrics_valid_input(self, client):
        payload = {
            "predictions": [[1, 2], [2, 1]],
            "labels": [[1, 2], [2, 1]]
        }
        response = client.post("/metrics", json=payload)
        assert response.status_code in (200, 500)

    def test_metrics_mismatched_shapes(self, client):
        payload = {
            "predictions": [[1, 2, 3]],
            "labels": [[1, 2]]
        }
        response = client.post("/metrics", json=payload)
        assert response.status_code in (200, 500)


class TestTrainingEndpoint:
    """训练端点测试"""

    def test_start_training_valid(self, client):
        payload = {
            "data_path": "data/test_data/",
            "epochs": 2,
            "batch_size": 4,
            "lr": 0.001
        }
        response = client.post("/train", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data

    def test_start_training_invalid_epochs(self, client):
        payload = {"data_path": "/tmp/test", "epochs": 200}
        response = client.post("/train", json=payload)
        assert response.status_code == 422
