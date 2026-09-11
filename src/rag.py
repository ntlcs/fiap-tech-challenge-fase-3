from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
    RAG_TOP_K
)


_embeddings = None
_vectorstore = None


def carregar_embeddings():
    """
    Carrega o modelo de embeddings utilizado
    na recuperação semântica dos protocolos.
    """

    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embeddings


def carregar_vectorstore():
    """
    Carrega o índice FAISS persistido em disco.
    """

    global _vectorstore

    if _vectorstore is None:
        embeddings = carregar_embeddings()

        _vectorstore = FAISS.load_local(
            str(VECTOR_DB_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    return _vectorstore


def buscar_protocolos(
    pergunta: str,
    k: int = RAG_TOP_K
):
    """
    Recupera os protocolos mais relevantes
    para a pergunta clínica.
    """

    vectorstore = carregar_vectorstore()

    return vectorstore.similarity_search(
        pergunta,
        k=k
    )


def formatar_contexto_protocolos(
    documentos
) -> str:
    """
    Formata os documentos recuperados para
    inclusão no prompt da LLM.
    """

    if not documentos:
        return ""

    blocos = []

    for doc in documentos:
        protocolo_id = doc.metadata.get(
            "protocolo_id",
            "N/A"
        )

        titulo = doc.metadata.get(
            "titulo",
            "Sem título"
        )

        blocos.append(
            (
                f"Protocolo: {protocolo_id}\n"
                f"Título: {titulo}\n"
                f"Conteúdo: {doc.page_content}"
            )
        )

    return "\n\n".join(blocos)


def extrair_fontes(
    documentos
) -> list[dict]:
    """
    Extrai metadados dos protocolos utilizados
    para rastreabilidade e auditoria.
    """

    fontes = []

    for doc in documentos:
        fontes.append(
            {
                "protocolo_id":
                    doc.metadata.get(
                        "protocolo_id"
                    ),
                "titulo":
                    doc.metadata.get(
                        "titulo"
                    )
            }
        )

    return fontes