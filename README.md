# Detecção de Pneumonia com Machine Learning

**POSTECH — IA para Devs — Tech Challenge Fase 1 — Desafio B**

Modelo de ML para classificação de diagnóstico clínico respiratório (pneumonia, edema pulmonar, atelectasia) a partir de dados clínicos tabulares.

---

## Ciclo ML utilizado

Este projeto segue as **7 etapas do ML Life Cycle**:

| Step | Etapa | Notebook/Arquivo |
|---|---|---|
| 1 — Collect the Data | Coleta e documentação do dataset | `notebooks/01_coleta_dados.ipynb` |
| 2 — Prepare the Data | EDA + Pré-processamento | `notebooks/02_eda.ipynb`, `notebooks/03_preprocessamento.ipynb` |
| 3 — Choose a Model | Escolha e justificativa dos algoritmos | `notebooks/04_escolha_modelo.ipynb` |
| 4 — Train the Model | Treinamento dos modelos base | `notebooks/05_treinamento.ipynb` |
| 5 — Parameter Tuning | GridSearchCV + Cross-Validation | `notebooks/06_tuning.ipynb` |
| 6 — Evaluation & Testing | Avaliação final + SHAP | `notebooks/07_avaliacao_shap.ipynb` |
| 7 — Deployment | API FastAPI + Docker | `app/main.py` |

---

## Estrutura do Projeto

```
repositorio/
├── data/                          ← datasets (NÃO vai pro git)
│   ├── tabular/                   ← CSV do dataset clínico
│   └── images/                    ← raio-X (parte extra)
├── notebooks/
│   ├── 01_coleta_dados.ipynb      ← Step 1: Collect
│   ├── 02_eda.ipynb               ← Step 2: Prepare (EDA)
│   ├── 03_preprocessamento.ipynb  ← Step 2: Prepare (pipeline)
│   ├── 04_escolha_modelo.ipynb    ← Step 3: Choose a Model
│   ├── 05_treinamento.ipynb       ← Step 4: Train
│   ├── 06_tuning.ipynb            ← Step 5: Parameter Tuning
│   ├── 07_avaliacao_shap.ipynb    ← Step 6: Evaluation
│   └── 08_cnn_imagens.ipynb       ← Extra: CNN
├── app/
│   └── main.py                    ← Step 7: FastAPI
├── models/                        ← modelos serializados (.pkl, .h5)
├── reports/                       ← gráficos e relatório técnico
├── Dockerfile
├── requirements.txt
└── .gitignore
```

---

## Datasets Utilizados

- **Tabular:** [Pneumonia Prediction Dataset](https://www.kaggle.com/datasets/ajithdari/pneumonia-prediction-dataset) — Ajith Dari — Licença: CC0 Public Domain
  - 1.500 registros, 297 pacientes, 3 classes (pneumonia, edema pulmonar, atelectasia)
  - Features: febre, taquicardia, estertores, saturação O₂, leucócitos, resultado raio-X
- **Imagens (extra):** [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — CC BY 4.0

---

## Como Executar

### Com Docker — Jupyter (análise)

```bash
docker build -t pneumonia-challenge .
docker run -p 8888:8888 pneumonia-challenge
# Abrir no navegador: http://localhost:8888
```

### Com Docker — API (serving)

```bash
docker run -p 8000:8000 pneumonia-challenge \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
# Documentação interativa: http://localhost:8000/docs
```

### Sem Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab          # para os notebooks
# ou
uvicorn app.main:app --reload   # para a API
```

---

## Resultados Obtidos

| Modelo | Accuracy | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Regressão Logística (base) | — | — | — | — |
| Regressão Logística (tuned) | — | — | — | — |
| Random Forest (base) | — | — | — | — |
| Random Forest (tuned) | — | — | — | — |
| CNN — raio-X (extra) | — | — | — | — |

*Tabela será preenchida ao final das etapas de treinamento e avaliação.*

---

## Tecnologias

Python 3.11, scikit-learn, SHAP, FastAPI, Jupyter, Docker, pandas, numpy, matplotlib, seaborn
