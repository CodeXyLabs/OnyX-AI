FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers>=4.44.0 \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]