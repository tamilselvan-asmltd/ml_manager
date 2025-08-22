FROM python:3.9-slim-buster

WORKDIR /workspace

COPY requirements.txt .
COPY train.py .
COPY params.json .

RUN python -m venv venv
RUN /workspace/venv/bin/pip install --no-cache-dir -r requirements.txt

CMD ["/workspace/venv/bin/python", "train.py"]