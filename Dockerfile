FROM python:3.7.1

WORKDIR /code
RUN apt-get update && apt-get install libgl1 -y && apt-get install libgdcm-tools -y
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

# USER root 
# RUN chmod 755 /pacs_connection/my_wrapper_script.sh
# CMD /pacs_connection/my_wrapper_script.sh

EXPOSE 7000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "2"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--reload"]

