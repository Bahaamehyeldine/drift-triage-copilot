FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY migrations/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Alembic configuration and migration scripts remain at the repository root.
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

CMD ["alembic", "-c", "/app/alembic.ini", "upgrade", "head"]