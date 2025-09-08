FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY app/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY README.md ./README.md
COPY LICENSE ./LICENSE

EXPOSE 5000

ENV WEIGHTS_PATH=/app/weights/resnet18_finetune_best.pt

CMD ["python", "app/app.py"]
