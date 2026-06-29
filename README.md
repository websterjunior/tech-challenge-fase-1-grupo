# FACULDADE DE INFORMÁTICA E ADMINISTRAÇÃO PAULISTA - FIAP
## PÓS TECH - IA PARA DEVS

### GRUPO 42:
 - Felipe Mendes Garcia
 - Silmara Alvares Barbosa
 - Webster Silva Campelo Junior
 - William Norio Yassuda
<br />

# Detecção de Pneumonia com Machine Learning

**POSTECH — IA para Devs — Tech Challenge Fase 1 — Desafio B**

Classificador de doenças respiratórias (pneumonia, edema pulmonar, atelectasia) a partir de dados clínicos tabulares, com extensão opcional de detecção via CNN em imagens de raio-X.

---

## Ciclo ML Utilizado

Este projeto segue as **7 etapas do ML Life Cycle**:

| Step | Etapa | Notebook / Arquivo |
|---|---|---|
| 1 — Collect the Data | Coleta e documentação do dataset | `notebooks/01_coleta_dados.ipynb` |
| 2 — Prepare the Data | EDA + Pré-processamento | `notebooks/02_eda.ipynb`, `notebooks/03_preprocessamento.ipynb` |
| 3 — Choose a Model | Escolha e justificativa dos algoritmos | `notebooks/04_escolha_modelo.ipynb` |
| 4 — Train the Model | Treinamento dos modelos base | `notebooks/05_treinamento.ipynb` |
| 5 — Parameter Tuning | GridSearchCV + Cross-Validation | `notebooks/06_tuning.ipynb` |
| 6 — Evaluation & Testing | Avaliação final + SHAP | `notebooks/07_avaliacao_shap.ipynb` |
| **7 — Deployment** | **API FastAPI + Docker** | **`app/main.py`** |
| Extra | CNN com Transfer Learning (raio-X) | `notebooks/08_cnn_imagens.ipynb` |

---

## Estrutura do Projeto

```
repositorio/
├── data/                              ← datasets (NÃO vai pro git)
│   ├── tabular/                       ← CSV do dataset clínico
│   └── images/chest_xray/            ← raio-X (parte extra)
├── notebooks/
│   ├── 01_coleta_dados.ipynb          ← Step 1: Collect
│   ├── 02_eda.ipynb                   ← Step 2: Prepare (EDA)
│   ├── 03_preprocessamento.ipynb      ← Step 2: Prepare (pipeline)
│   ├── 04_escolha_modelo.ipynb        ← Step 3: Choose a Model
│   ├── 05_treinamento.ipynb           ← Step 4: Train
│   ├── 06_tuning.ipynb                ← Step 5: Parameter Tuning
│   ├── 07_avaliacao_shap.ipynb        ← Step 6: Evaluation & Testing
│   └── 08_cnn_imagens.ipynb           ← Extra: CNN com Transfer Learning
├── app/
│   └── main.py                        ← Step 7: API FastAPI
├── models/                            ← artefatos serializados (.pkl, .keras)
│   ├── melhor_modelo_tuned.pkl        ← modelo final (Regressão Logística)
│   ├── scaler.pkl                     ← StandardScaler (oxygen_saturation, wbc_count)
│   ├── label_encoder.pkl              ← LabelEncoder das classes
│   ├── cnn_finetuned.keras            ← CNN fine-tuned (MobileNetV2)
│   └── ...                            ← modelos base e checkpoints
├── reports/
│   ├── relatorio_tecnico.md           ← relatório técnico completo
│   └── *.png                          ← gráficos gerados pelos notebooks
├── Dockerfile
├── requirements.txt
└── .gitignore
```

---

## Datasets Utilizados

### Parte Obrigatória — Dados Tabulares
- **Dataset:** [Pneumonia Prediction Dataset](https://www.kaggle.com/datasets/ajithdari/pneumonia-prediction-dataset) — Ajith Dari
- **Licença:** CC0 Public Domain
- **Características:** 1.500 registros, 3 classes balanceadas (pneumonia, edema pulmonar, atelectasia)
- **Features clínicas:** febre, taquicardia, estertores, saturação O₂, leucócitos, resultado raio-X

> O dataset não está incluso no repositório. Baixar via Kaggle CLI:
> ```bash
> kaggle datasets download -d ajithdari/pneumonia-prediction-dataset \
>   -p data/tabular/ --unzip
> ```

### Parte Extra — Imagens de Raio-X
- **Dataset:** [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Paul Mooney
- **Licença:** CC BY 4.0 (Kermany, Zhang, Goldbaum — 2018)
- **Características:** 5.856 imagens JPEG, 2 classes (NORMAL, PNEUMONIA), pré-dividido em train/val/test

> ```bash
> kaggle datasets download -d paultimothymooney/chest-xray-pneumonia \
>   -p data/images/ --unzip
> ```

---

## Como Executar

### Com Docker — Jupyter (análise dos notebooks)

```bash
docker build -t pneumonia-challenge .
docker run -p 8888:8888 pneumonia-challenge
# Abrir no navegador: http://localhost:8888
```

### Com Docker — API (serving do modelo)

```bash
docker build -t pneumonia-challenge .
docker run -p 8000:8000 pneumonia-challenge \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
# Documentação interativa: http://localhost:8000/docs
```

### Sem Docker

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

# Instalar dependências
pip install -r requirements.txt

# Opção 1: Jupyter (análise/notebooks)
jupyter lab

# Opção 2: API de serving
uvicorn app.main:app --reload --port 8000
# Acessar: http://localhost:8000/docs
```

### Testar a API

Com a API rodando, use o Swagger em `http://localhost:8000/docs` ou via `curl`:

```bash
curl -X POST http://localhost:8000/predizer \
  -H "Content-Type: application/json" \
  -d '{
    "fever": 1,
    "tachycardia": 1,
    "crackles": 1,
    "oxygen_saturation": 92.5,
    "wbc_count": 11.2,
    "resultado_raio_x": "infiltrate"
  }'
```

Resposta esperada:
```json
{
  "diagnostico": "pneumonia",
  "probabilidade_diagnostico": 0.4008,
  "probabilidades": {
    "atelectasis": 0.2708,
    "pneumonia": 0.4008,
    "pulmonary_edema": 0.3284
  },
  "alerta": "RISCO MODERADO-ALTO de pneumonia (40%). Recomenda-se avaliação médica imediata...",
  "aviso": "Sistema de suporte ao diagnóstico. A decisão clínica é responsabilidade exclusiva do profissional de saúde habilitado."
}
```

---

## Resultados Obtidos

### Parte Obrigatória — Dados Tabulares

**Conjunto de validação — comparativo dos modelos:**

| Modelo | Accuracy | Recall Macro | F1 Macro |
|---|---|---|---|
| Regressão Logística (base) | 50,7% | **50,7%** | 49,0% |
| Árvore de Decisão (base) | 39,6% | 39,6% | 37,1% |
| Random Forest (base) | 35,1% | 35,1% | 34,8% |
| KNN (base) | 33,3% | 33,3% | 16,7% |
| **Regressão Logística (tuned)** | **50,7%** | **50,7%** | **49,0%** |

**Avaliação final no conjunto de teste (modelo tuned — primeira e única vez):**

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| atelectasis | 0,00 | 0,00 | 0,00 |
| pneumonia | 0,00 | 0,00 | 0,00 |
| pulmonary edema | 0,33 | 1,00 | 0,50 |
| **macro avg** | **0,11** | **0,33** | **0,17** |

| Métrica | Valor |
|---|---|
| Accuracy | 33,3% |
| AUC (pneumonia) | 0,626 |

> **Análise crítica:** O modelo colapsa para predição exclusiva de `pulmonary edema`,
> atingindo accuracy equivalente ao acaso. As features clínicas disponíveis são insuficientes
> para discriminar as 3 condições (ver `reports/relatorio_tecnico.md`).
> O valor educacional está no processo completo do ML Life Cycle, na aplicação de SHAP
> para interpretabilidade e na discussão crítica sobre limitações.

### Parte Extra — CNN com Imagens de Raio-X

| Modelo | Dataset | Arquitetura | Fase |
|---|---|---|---|
| CNN Fase 1 | Chest X-Ray (5.856 imgs) | MobileNetV2 (base congelada) | Transfer Learning |
| **CNN Fase 2** | Chest X-Ray (5.856 imgs) | MobileNetV2 (last 20 layers) | Fine-Tuning |

> Métricas de avaliação da CNN no conjunto de teste disponíveis após execução do
> notebook `08_cnn_imagens.ipynb`.

---

## Tecnologias

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.14 |
| Análise de Dados | pandas, numpy |
| Visualização | matplotlib, seaborn |
| ML Clássico | scikit-learn (GridSearchCV, cross_val_score) |
| Explicabilidade | SHAP (TreeExplainer, summary_plot, waterfall) |
| Visão Computacional | Keras 3 + PyTorch backend (MobileNetV2) |
| API de Serving | FastAPI, uvicorn, Pydantic |
| Ambiente | Jupyter Lab, Docker |
| Versionamento | Git + GitHub |

---

## Relatório Técnico

O relatório técnico completo — cobrindo todas as 6 seções do ML Life Cycle — está em:
[`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md)
