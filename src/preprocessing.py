import pandas as pd


COLUNAS_PII = [
    "nome",
    "cpf",
    "email",
    "telefone"
]


def anonimizar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas contendo informações
    pessoalmente identificáveis (PII).
    """

    df_anonimizado = df.copy()

    colunas_encontradas = [
        coluna
        for coluna in COLUNAS_PII
        if coluna in df_anonimizado.columns
    ]

    return df_anonimizado.drop(
        columns=colunas_encontradas
    )


def normalizar_dados(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Trata valores ausentes, normaliza campos
    textuais e remove registros duplicados.
    """

    df_processado = df.copy()

    if "exame_pendente" in df_processado.columns:
        df_processado["exame_pendente"] = (
            df_processado["exame_pendente"]
            .fillna("Sem exame pendente")
        )

    colunas_texto = [
        "sexo",
        "diagnostico",
        "exame_pendente"
    ]

    for coluna in colunas_texto:
        if coluna in df_processado.columns:
            df_processado[coluna] = (
                df_processado[coluna]
                .astype(str)
                .str.strip()
            )

    df_processado = (
        df_processado
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_processado


def validar_registros(
    df: pd.DataFrame
) -> pd.Series:
    """
    Valida intervalos de plausibilidade dos
    principais dados clínicos.
    """

    validacoes = {
        "idade_valida":
            df["idade"].between(0, 120),

        "glicemia_valida":
            df["glicemia_mg_dl"].between(
                20,
                1000
            ),

        "hba1c_valida":
            df["hba1c_percentual"].between(
                2,
                20
            ),

        "pressao_sistolica_valida":
            df["pressao_sistolica"].between(
                50,
                300
            ),

        "pressao_diastolica_valida":
            df["pressao_diastolica"].between(
                30,
                200
            ),

        "imc_valido":
            df["imc"].between(
                10,
                80
            ),

        "creatinina_valida":
            df["creatinina_mg_dl"].between(
                0.1,
                20
            )
    }

    resultado_validacao = pd.DataFrame(
        validacoes,
        index=df.index
    )

    return resultado_validacao.all(axis=1)


def curar_dados(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Mantém apenas registros considerados
    válidos segundo as regras de curadoria.
    """

    registros_validos = validar_registros(df)

    return (
        df.loc[registros_validos]
        .reset_index(drop=True)
    )


def preprocessar_dados(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Executa o pipeline completo:
    anonimização, normalização e curadoria.
    """

    df_anonimizado = anonimizar_dados(df)

    df_normalizado = normalizar_dados(
        df_anonimizado
    )

    df_curado = curar_dados(
        df_normalizado
    )

    return df_curado