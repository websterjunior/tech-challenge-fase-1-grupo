"""
API de Serving — Detecção de Doença Respiratória
Step 7 do ML Life Cycle: Deployment & Forecasting

Modelo: Regressão Logística (tuned) — 3 classes:
  0 = atelectasis | 1 = pneumonia | 2 = pulmonary edema
"""

import logging
import pickle
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ── Configuração de logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models"

FEATURE_ORDER = [
    "fever",
    "tachycardia",
    "crackles",
    "oxygen_saturation",
    "wbc_count",
    "chest_xray_result_effusion",
    "chest_xray_result_infiltrate",
    "chest_xray_result_normal",
    "chest_xray_result_opacity",
]

COLUNAS_NUMERICAS = ["oxygen_saturation", "wbc_count"]

DIAGNOSTICOS = {
    0: "atelectasis",
    1: "pneumonia",
    2: "pulmonary edema",
}

# ── Carregamento de artefatos na inicialização ───────────────────────────────
def _carregar_artefatos() -> tuple:
    try:
        with open(MODEL_PATH / "melhor_modelo_tuned.pkl", "rb") as f:
            modelo = pickle.load(f)
        with open(MODEL_PATH / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_PATH / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("Artefatos carregados: modelo=%s", type(modelo).__name__)
        return modelo, scaler, label_encoder
    except FileNotFoundError as e:
        logger.error("Artefato não encontrado: %s", e)
        raise RuntimeError(f"Artefato não encontrado: {e}") from e


modelo, scaler, label_encoder = _carregar_artefatos()

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Detecção de Doença Respiratória — API",
    description=(
        "Classificação clínica de doenças respiratórias (atelectasia, pneumonia, edema pulmonar) "
        "a partir de dados clínicos. "
        "**Atenção:** este sistema é um suporte ao diagnóstico — não substitui avaliação médica."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ResultadoRaioX(str, Enum):
    consolidation = "consolidation"
    effusion = "effusion"
    infiltrate = "infiltrate"
    normal = "normal"
    opacity = "opacity"


class DadosClinicos(BaseModel):
    """Features clínicas para classificação de doença respiratória."""

    fever: int = Field(..., ge=0, le=1, description="Febre presente (1) ou ausente (0)")
    tachycardia: int = Field(..., ge=0, le=1, description="Taquicardia presente (1) ou ausente (0)")
    crackles: int = Field(..., ge=0, le=1, description="Estertores (crepitações) presentes (1) ou ausentes (0)")
    oxygen_saturation: float = Field(
        ..., ge=70.0, le=100.0, description="Saturação de oxigênio (%)"
    )
    wbc_count: float = Field(
        ..., ge=1.0, le=50.0, description="Contagem de leucócitos (10³/µL)"
    )
    resultado_raio_x: ResultadoRaioX = Field(
        ...,
        description=(
            "Achado radiológico: "
            "'consolidation' | 'effusion' | 'infiltrate' | 'normal' | 'opacity'"
        ),
    )

    @field_validator("oxygen_saturation")
    @classmethod
    def validar_saturacao(cls, v: float) -> float:
        if v < 80.0:
            logger.warning("Saturação muito baixa (%.1f%%) — possível emergência clínica", v)
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fever": 1,
                    "tachycardia": 1,
                    "crackles": 1,
                    "oxygen_saturation": 92.5,
                    "wbc_count": 11.2,
                    "resultado_raio_x": "infiltrate",
                }
            ]
        }
    }


class Probabilidades(BaseModel):
    atelectasis: float
    pneumonia: float
    pulmonary_edema: float


class Predicao(BaseModel):
    diagnostico: str = Field(..., description="Classe com maior probabilidade")
    probabilidade_diagnostico: float = Field(..., description="Probabilidade da classe predita (0 a 1)")
    probabilidades: Probabilidades = Field(..., description="Probabilidades para as 3 classes")
    alerta: str = Field(..., description="Mensagem clínica de orientação")
    aviso: str = Field(
        default=(
            "Sistema de suporte ao diagnóstico. "
            "A decisão clínica é responsabilidade exclusiva do profissional de saúde habilitado."
        )
    )


# ── Lógica de predição ────────────────────────────────────────────────────────
def _one_hot_raio_x(resultado: str) -> dict:
    """Converte o resultado do raio-X em variáveis dummy (consolidation = base)."""
    categorias = ["effusion", "infiltrate", "normal", "opacity"]
    return {f"chest_xray_result_{cat}": int(resultado == cat) for cat in categorias}


def _montar_entrada(dados: DadosClinicos) -> pd.DataFrame:
    """Monta o DataFrame de entrada com as features na ordem esperada pelo modelo."""
    raio_x_dummies = _one_hot_raio_x(dados.resultado_raio_x.value)

    registro = {
        "fever": dados.fever,
        "tachycardia": dados.tachycardia,
        "crackles": dados.crackles,
        "oxygen_saturation": dados.oxygen_saturation,
        "wbc_count": dados.wbc_count,
        **raio_x_dummies,
    }

    df = pd.DataFrame([registro], columns=FEATURE_ORDER)
    df[COLUNAS_NUMERICAS] = scaler.transform(df[COLUNAS_NUMERICAS])
    return df


def _gerar_alerta(diagnostico: str, prob: float) -> str:
    alertas = {
        "pneumonia": (
            f"RISCO MODERADO-ALTO de pneumonia ({prob:.0%}). "
            "Recomenda-se avaliação médica imediata, hemograma e radiografia confirmatória."
        ),
        "pulmonary edema": (
            f"RISCO MODERADO-ALTO de edema pulmonar ({prob:.0%}). "
            "Monitorar saturação de O₂. Avaliação cardiológica recomendada."
        ),
        "atelectasis": (
            f"Padrão compatível com atelectasia ({prob:.0%}). "
            "Fisiioterapia respiratória e acompanhamento clínico indicados."
        ),
    }
    return alertas.get(diagnostico, "Avaliação clínica complementar recomendada.")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predizer", response_model=Predicao, summary="Classificar doença respiratória")
def predizer(dados: DadosClinicos) -> Predicao:
    """
    Recebe dados clínicos e retorna o diagnóstico diferencial entre:
    - **atelectasis** — atelectasia
    - **pneumonia** — pneumonia
    - **pulmonary edema** — edema pulmonar

    > ⚠️ Este endpoint é um suporte à decisão clínica, não um diagnóstico definitivo.
    """
    try:
        df_entrada = _montar_entrada(dados)
        proba = modelo.predict_proba(df_entrada)[0]
        classe_idx = int(np.argmax(proba))
        diagnostico = label_encoder.classes_[classe_idx]
        prob_pred = float(proba[classe_idx])

        logger.info(
            "Predição: %s (%.2f%%) | febre=%d crackles=%d SpO2=%.1f WBC=%.1f raio_x=%s",
            diagnostico, prob_pred * 100,
            dados.fever, dados.crackles,
            dados.oxygen_saturation, dados.wbc_count,
            dados.resultado_raio_x.value,
        )

        return Predicao(
            diagnostico=diagnostico,
            probabilidade_diagnostico=round(prob_pred, 4),
            probabilidades=Probabilidades(
                atelectasis=round(float(proba[0]), 4),
                pneumonia=round(float(proba[1]), 4),
                pulmonary_edema=round(float(proba[2]), 4),
            ),
            alerta=_gerar_alerta(diagnostico, prob_pred),
        )

    except Exception as e:
        logger.error("Erro na predição: %s", e)
        raise HTTPException(status_code=500, detail=f"Erro interno na predição: {e}") from e


@app.get("/saude", summary="Health check")
def saude() -> dict:
    """Verifica se a API está operacional."""
    return {
        "status": "ok",
        "modelo": type(modelo).__name__,
        "classes": label_encoder.classes_.tolist(),
        "versao": "1.0.0",
    }


@app.get("/", include_in_schema=False)
def raiz() -> dict:
    return {"mensagem": "API operacional. Acesse /docs para a documentação interativa."}
