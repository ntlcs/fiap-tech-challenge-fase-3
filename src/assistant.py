from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.llm import gerar_resposta


TEMPLATE_CLINICO = """
### Instrução:

Você é um assistente virtual de apoio clínico.

Utilize SOMENTE as informações dos dados do paciente
e dos protocolos institucionais fornecidos.

Regras obrigatórias:

1. Não prescreva medicamentos.
2. Não informe doses.
3. Não altere tratamento.
4. Não invente informações clínicas.
5. Não faça diagnóstico novo.
6. Se a informação não estiver disponível, informe que não há dados suficientes.
7. Toda decisão clínica deve ser validada pelo médico responsável.
8. Informe as fontes utilizadas.

### Pergunta:
{pergunta}

### Dados do paciente:
{contexto_paciente}

### Protocolos institucionais:
{contexto_protocolos}

### Resposta:
"""


prompt_clinico = PromptTemplate.from_template(
    TEMPLATE_CLINICO
)


llm_runnable = RunnableLambda(
    lambda prompt_value: gerar_resposta(
        prompt_value.to_string()
    )
)


clinical_chain = (
    prompt_clinico
    | llm_runnable
)


def gerar_resposta_clinica(
    pergunta: str,
    contexto_paciente: str,
    contexto_protocolos: str
) -> str:
    """
    Executa a cadeia clínica utilizando
    LangChain + modelo fine-tuned.
    """

    return clinical_chain.invoke(
        {
            "pergunta": pergunta,
            "contexto_paciente":
                contexto_paciente,
            "contexto_protocolos":
                contexto_protocolos
        }
    )