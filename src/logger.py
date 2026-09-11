import json
import uuid

from datetime import (
    datetime,
    timezone
)

from src.config import (
    LOG_PATH,
    LOGS_DIR
)


def registrar_auditoria(
    state: dict
) -> dict:
    """
    Registra uma execução do assistente clínico
    em arquivo JSONL para fins de auditoria
    e rastreabilidade.

    Retorna o identificador único da execução.
    """

    audit_id = str(
        uuid.uuid4()
    )

    registro = {
        "audit_id": audit_id,

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "patient_id": state.get(
            "patient_id"
        ),

        "pergunta": state.get(
            "pergunta"
        ),

        "resposta_llm": state.get(
            "resposta_llm"
        ),

        "resposta_final": state.get(
            "resposta_final"
        ),

        "entrada_bloqueada": state.get(
            "entrada_bloqueada",
            False
        ),

        "saida_bloqueada": state.get(
            "saida_bloqueada",
            False
        ),

        "motivos_seguranca": state.get(
            "motivos_seguranca",
            []
        ),

        "fontes": state.get(
            "fontes",
            []
        ),

        "human_validation_required": state.get(
            "human_validation_required",
            True
        )
    }

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        LOG_PATH,
        "a",
        encoding="utf-8"
    ) as arquivo:
        arquivo.write(
            json.dumps(
                registro,
                ensure_ascii=False
            )
            + "\n"
        )

    return {
        "audit_id": audit_id
    }