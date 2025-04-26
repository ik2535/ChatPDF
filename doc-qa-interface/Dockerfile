FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY query_app.py ./
COPY retrieve.py ./
CMD ["python", "query_app.py"]
