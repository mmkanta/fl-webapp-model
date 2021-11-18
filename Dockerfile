FROM python:3.7.1

WORKDIR /code
RUN apt-get update && apt-get install libgl1 -y
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]

# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# WORKDIR /code
# RUN apt-get update && apt-get install libgl1 -y
# COPY requirements.txt requirements.txt
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
# COPY . .
# EXPOSE 8000:8000
