FROM python:3.9.17

COPY ./requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./finmodulo.py /finmodulo.py

EXPOSE 9000

CMD ["python", "finmodulo.py"]
