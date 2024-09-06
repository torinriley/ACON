FROM python:3.9-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y build-essential

COPY . .

RUN pip install --upgrade pip setuptools wheel twine

RUN python setup.py sdist bdist_wheel

CMD ["python", "-m", "unittest", "discover"]

