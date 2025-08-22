FROM python:3.9-slim-buster

WORKDIR /workspace

COPY linear_model linear_model

RUN python -m venv venv
RUN /workspace/venv/bin/pip install --no-cache-dir -r linear_model/requirements.txt

CMD ["/workspace/venv/bin/python", "linear_model/train.py"]