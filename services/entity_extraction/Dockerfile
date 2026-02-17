# services/entity-extraction/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && printf "Cython<3\n" > /tmp/pip-constraints.txt \
    && PIP_CONSTRAINT=/tmp/pip-constraints.txt pip install --no-cache-dir -r requirements.txt

# Скачивание DeepPavlov модели (опционально, можно делать при первом запуске)
RUN python -m deeppavlov install ner_rus_bert

# Копирование кода
COPY . .

# Expose порт
EXPOSE 5001

# Команда запуска
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
