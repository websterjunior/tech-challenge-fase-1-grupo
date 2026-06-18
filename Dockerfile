# ── Build da imagem ────────────────────────────────────────────────────────────
# Para análise/notebooks (Jupyter):
#   docker build -t pneumonia-challenge .
#   docker run -p 8888:8888 pneumonia-challenge
#   Abrir: http://localhost:8888
#
# Para API de serving (FastAPI):
#   docker build -t pneumonia-challenge .
#   docker run -p 8000:8000 pneumonia-challenge \
#     uvicorn app.main:app --host 0.0.0.0 --port 8000
#   Documentação: http://localhost:8000/docs
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema necessárias para bibliotecas ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
# Nota: torch/keras do requirements.txt são usados apenas nos notebooks (CNN).
# A API (app/main.py) usa somente scikit-learn, fastapi, uvicorn, pandas, numpy.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    scikit-learn \
    pandas \
    numpy \
    pydantic \
    && pip install --no-cache-dir jupyterlab matplotlib seaborn shap

# Copiar o projeto
COPY . .

# Portas expostas
EXPOSE 8888
EXPOSE 8000

# Modo padrão: Jupyter (análise dos notebooks)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
