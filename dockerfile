FROM python:3.12-slim

WORKDIR /revenue-recovery/

ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python models/viability_model.py

CMD ["python", "main.py"]