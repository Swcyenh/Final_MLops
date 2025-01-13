# Sử dụng image Python nhẹ
FROM python:3.11-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Cài đặt các thư viện yêu cầu
RUN pip install --no-cache-dir -r requirements.txt

# Expose port nếu cần (ví dụ, nếu ứng dụng của bạn chạy server)
EXPOSE 8080

# Chạy ứng dụng
CMD ["python", "main.py"]
