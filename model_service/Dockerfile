FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY model_service/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY model_service ./model_service
COPY shared ./shared

EXPOSE 8000

CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0", "--port", "8000"]