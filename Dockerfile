FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main

COPY . .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]