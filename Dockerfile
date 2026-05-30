FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888
EXPOSE 8000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]