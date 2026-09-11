import sqlite3
from typing import Optional

from src.config import DB_PATH


def buscar_paciente(patient_id: str) -> Optional[dict]:
    """
    Busca um paciente pelo identificador.

    Retorna:
        dict com os dados do paciente, se encontrado.
        None, caso o paciente não exista.
    """

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(
            """
            SELECT *
            FROM pacientes
            WHERE patient_id = ?
            """,
            (patient_id,)
        )

        registro = cursor.fetchone()

    if registro is None:
        return None

    return dict(registro)


def listar_pacientes() -> list[dict]:
    """
    Retorna todos os pacientes cadastrados.
    """

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(
            """
            SELECT *
            FROM pacientes
            ORDER BY patient_id
            """
        )

        registros = cursor.fetchall()

    return [
        dict(registro)
        for registro in registros
    ]


def formatar_contexto_paciente(
    paciente: dict | None
) -> str:
    """
    Converte os dados estruturados do paciente
    em texto para utilização no contexto da LLM.
    """

    if paciente is None:
        return ""

    linhas = []

    for campo, valor in paciente.items():
        linhas.append(
            f"{campo}: {valor}"
        )

    return "\n".join(linhas)