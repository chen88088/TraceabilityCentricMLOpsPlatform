# 使用 Python 3.10 作為基底映像
FROM python:3.10

# 設定工作目錄
WORKDIR /app

# 複製當前目錄下的檔案到容器內
COPY . .

# 安裝依賴套件
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 暴露 FastAPI 埠號
EXPOSE 8088

# 啟動 FastAPI 應用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088", "--reload"]
