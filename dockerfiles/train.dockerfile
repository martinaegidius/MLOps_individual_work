# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY configs/ configs/
COPY src/ src/
# COPY data/ data/
# COPY reports/ reports/
# COPY models/ models/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install torch~=2.5.1 -i https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt --no-cache-dir --verbose 
#install dependencies from local pip cache
# RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/mlops_individual/train.py"]
