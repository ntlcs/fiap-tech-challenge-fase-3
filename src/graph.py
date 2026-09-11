from typing import (
    Any,
    TypedDict
)

from langgraph.graph import (
    END,
    START,
    StateGraph
)

from src.assistant import gerar_resposta_clinica
from src.database import (
    buscar_paciente,
    formatar_contexto_paciente
)
from src.logger import registrar_auditoria
from src.rag import (
    buscar_protocolos,
    extrair_fontes,
    formatar_contexto_protocolos
)
from src.safety import (
    validar_entrada,
    validar_saida
)


# ============================================================
# Estado compartilhado
# ============================================================

class ClinicalState(
    TypedDict,
    total=False
):
    patient_id: str
    pergunta: str

    paciente: dict | None
    contexto_paciente: str

    documentos: list[Any]
    contexto_protocolos: str

    resposta_llm: str
    resposta_final: str

    entrada_bloqueada: bool
    saida_bloqueada: bool
    motivos_seguranca: list[str]

    fontes: list[dict]

    human_validation_required: bool
    audit_id: str


# ============================================================
# Nós
# ============================================================

def node_validar_entrada(
    state: ClinicalState
):
    resultado = validar_entrada(
        state["pergunta"]
    )

    return {
        "entrada_bloqueada":
            resultado["bloqueada"],

        "motivos_seguranca":
            resultado["motivos"]
    }


def node_bloquear_entrada(
    state: ClinicalState
):
    return {
        "resposta_final": (
            "Não posso indicar medicamento, "
            "dose ou alteração de tratamento. "
            "Posso apresentar dados do paciente "
            "e protocolos institucionais para "
            "apoio à avaliação do médico responsável."
        ),
        "human_validation_required": True
    }


def node_buscar_paciente(
    state: ClinicalState
):
    paciente = buscar_paciente(
        state["patient_id"]
    )

    if paciente is None:
        return {
            "paciente": None,
            "contexto_paciente": ""
        }

    return {
        "paciente": paciente,

        "contexto_paciente":
            formatar_contexto_paciente(
                paciente
            )
    }


def node_paciente_nao_encontrado(
    state: ClinicalState
):
    return {
        "resposta_final": (
            "Paciente não encontrado na base "
            "de dados. Não foi realizada "
            "geração de resposta clínica."
        ),
        "human_validation_required": True
    }


def node_buscar_protocolos(
    state: ClinicalState
):
    documentos = buscar_protocolos(
        state["pergunta"]
    )

    return {
        "documentos": documentos,

        "contexto_protocolos":
            formatar_contexto_protocolos(
                documentos
            ),

        "fontes":
            extrair_fontes(
                documentos
            )
    }


def node_gerar_resposta(
    state: ClinicalState
):
    resposta = gerar_resposta_clinica(
        pergunta=state["pergunta"],

        contexto_paciente=state[
            "contexto_paciente"
        ],

        contexto_protocolos=state[
            "contexto_protocolos"
        ]
    )

    return {
        "resposta_llm": resposta
    }


def node_validar_saida(
    state: ClinicalState
):
    resultado = validar_saida(
        state["resposta_llm"]
    )

    motivos_anteriores = state.get(
        "motivos_seguranca",
        []
    )

    return {
        "saida_bloqueada":
            resultado["bloqueada"],

        "motivos_seguranca":
            motivos_anteriores
            + resultado["motivos"]
    }


def node_bloquear_saida(
    state: ClinicalState
):
    return {
        "resposta_final": (
            "A resposta gerada automaticamente "
            "foi bloqueada pela camada de segurança "
            "por conter conteúdo clínico não "
            "suportado ou fora do contexto autorizado. "
            "A avaliação deve ser realizada pelo "
            "médico responsável."
        ),
        "human_validation_required": True
    }


def node_resposta_segura(
    state: ClinicalState
):
    resposta = state["resposta_llm"]

    fontes = state.get(
        "fontes",
        []
    )

    if fontes:
        texto_fontes = "\n".join(
            [
                (
                    f"- {fonte['protocolo_id']}: "
                    f"{fonte['titulo']}"
                )
                for fonte in fontes
            ]
        )

        resposta += (
            "\n\nFontes institucionais utilizadas:\n"
            + texto_fontes
        )

    resposta += (
        "\n\nValidação humana obrigatória "
        "antes de qualquer decisão clínica."
    )

    return {
        "resposta_final": resposta,
        "human_validation_required": True
    }


def node_auditoria(
    state: ClinicalState
):
    return registrar_auditoria(
        state
    )


# ============================================================
# Rotas condicionais
# ============================================================

def rota_apos_validar_entrada(
    state: ClinicalState
):
    if state.get(
        "entrada_bloqueada",
        False
    ):
        return "bloquear_entrada"

    return "buscar_paciente"


def rota_apos_buscar_paciente(
    state: ClinicalState
):
    if state.get("paciente") is None:
        return "paciente_nao_encontrado"

    return "buscar_protocolos"


def rota_apos_validar_saida(
    state: ClinicalState
):
    if state.get(
        "saida_bloqueada",
        False
    ):
        return "bloquear_saida"

    return "resposta_segura"


# ============================================================
# Construção do LangGraph
# ============================================================

def criar_grafo_clinico():
    graph_builder = StateGraph(
        ClinicalState
    )

    graph_builder.add_node(
        "validar_entrada",
        node_validar_entrada
    )

    graph_builder.add_node(
        "bloquear_entrada",
        node_bloquear_entrada
    )

    graph_builder.add_node(
        "buscar_paciente",
        node_buscar_paciente
    )

    graph_builder.add_node(
        "paciente_nao_encontrado",
        node_paciente_nao_encontrado
    )

    graph_builder.add_node(
        "buscar_protocolos",
        node_buscar_protocolos
    )

    graph_builder.add_node(
        "gerar_resposta",
        node_gerar_resposta
    )

    graph_builder.add_node(
        "validar_saida",
        node_validar_saida
    )

    graph_builder.add_node(
        "bloquear_saida",
        node_bloquear_saida
    )

    graph_builder.add_node(
        "resposta_segura",
        node_resposta_segura
    )

    graph_builder.add_node(
        "auditoria",
        node_auditoria
    )

    graph_builder.add_edge(
        START,
        "validar_entrada"
    )

    graph_builder.add_conditional_edges(
        "validar_entrada",
        rota_apos_validar_entrada,
        {
            "bloquear_entrada":
                "bloquear_entrada",

            "buscar_paciente":
                "buscar_paciente"
        }
    )

    graph_builder.add_conditional_edges(
        "buscar_paciente",
        rota_apos_buscar_paciente,
        {
            "paciente_nao_encontrado":
                "paciente_nao_encontrado",

            "buscar_protocolos":
                "buscar_protocolos"
        }
    )

    graph_builder.add_edge(
        "buscar_protocolos",
        "gerar_resposta"
    )

    graph_builder.add_edge(
        "gerar_resposta",
        "validar_saida"
    )

    graph_builder.add_conditional_edges(
        "validar_saida",
        rota_apos_validar_saida,
        {
            "bloquear_saida":
                "bloquear_saida",

            "resposta_segura":
                "resposta_segura"
        }
    )

    graph_builder.add_edge(
        "bloquear_entrada",
        "auditoria"
    )

    graph_builder.add_edge(
        "paciente_nao_encontrado",
        "auditoria"
    )

    graph_builder.add_edge(
        "bloquear_saida",
        "auditoria"
    )

    graph_builder.add_edge(
        "resposta_segura",
        "auditoria"
    )

    graph_builder.add_edge(
        "auditoria",
        END
    )

    return graph_builder.compile()


clinical_graph = criar_grafo_clinico()