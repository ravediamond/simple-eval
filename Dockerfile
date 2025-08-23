FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8080}"]