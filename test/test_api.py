import pytest
from api.main import app  # Import ứng dụng Flask của bạn
from flask import json

@pytest.fixture
def client():
    """Fixture to create a test client for Flask"""
    with app.test_client() as client:
        yield client

def test_health(client):
    """Kiểm tra route health"""
    response = client.get('/health')
    assert response.status_code == 200
    assert b'OK' in response.data

def test_predict(client):
    """Kiểm tra route dự đoán"""
    data = {
        "image": (open("test_image.jpg", "rb"), "test_image.jpg")
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b'predictions' in response.data

def test_error(client):
    """Kiểm tra trường hợp lỗi"""
    response = client.get('/non_existent_route')
    assert response.status_code == 404
