import re


PADROES_ENTRADA_PROIBIDA = {
    "prescricao": (
        r"\b(prescrev\w*|prescriç\w*|receit\w*)\b"
    ),
    "dose": (
        r"\b(dose|dosagem|quantos?\s*mg)\b"
    )
}


PADROES_SAIDA_PROIBIDA = {
    "prescricao_medicamento": (
        r"\b(deve|recomendo|indico|iniciar|administrar)\b"
        r".{0,50}"
        r"\b(metformina|insulina|lisinopril|medicamento|fármaco)\b"
    ),

    "dose_medicamento": (
        r"\b\d+(?:[.,]\d+)?\s*mg\b"
        r"(?!\s*/\s*d[lL])"
    ),

    "dialogo_fora_contexto": (
        r"(Human:|Assistant:)"
    ),

    "avaliacao_clinica_nao_suportada": (
        r"\b("
        r"adequad[ao]|"
        r"inadequad[ao]|"
        r"normal|"
        r"anormal|"
        r"recomendável|"
        r"ideal|"
        r"acima do limite|"
        r"abaixo do limite"
        r")\b"
    )
}


def validar_entrada(
    pergunta: str
) -> dict:
    """
    Valida se a pergunta contém uma solicitação
    proibida, como prescrição ou dosagem.
    """

    motivos = []

    for nome, padrao in (
        PADROES_ENTRADA_PROIBIDA.items()
    ):
        if re.search(
            padrao,
            pergunta,
            flags=re.IGNORECASE
        ):
            motivos.append(nome)

    return {
        "bloqueada": len(motivos) > 0,
        "motivos": motivos
    }


def validar_saida(
    resposta: str
) -> dict:
    """
    Valida se a resposta gerada pela LLM contém
    conteúdo considerado inseguro ou não suportado.
    """

    motivos = []

    for nome, padrao in (
        PADROES_SAIDA_PROIBIDA.items()
    ):
        if re.search(
            padrao,
            resposta,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            )
        ):
            motivos.append(nome)

    return {
        "bloqueada": len(motivos) > 0,
        "motivos": motivos
    }