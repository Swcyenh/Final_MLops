from flask import Flask

def create_app():
    """Khởi tạo ứng dụng Flask."""
    app = Flask(__name__)
    
    # Cấu hình có thể thêm vào nếu cần
    # app.config['SECRET_KEY'] = 'secret_key_example'
    
    return app
