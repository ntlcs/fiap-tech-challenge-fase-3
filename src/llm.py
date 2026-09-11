import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from src.config import (
    BASE_MODEL,
    MODEL_DIR,
    MAX_NEW_TOKENS
)


_tokenizer = None
_modelo = None


def carregar_modelo():
    """
    Carrega o tokenizer, o modelo base Qwen
    e o adapter LoRA treinado no projeto.

    O carregamento ocorre apenas uma vez.
    """

    global _tokenizer
    global _modelo

    if _tokenizer is not None and _modelo is not None:
        return _tokenizer, _modelo

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = (
            _tokenizer.eos_token
        )

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    modelo_base = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            device_map="auto"
        )
    )

    _modelo = PeftModel.from_pretrained(
        modelo_base,
        str(MODEL_DIR)
    )

    _modelo.eval()

    return _tokenizer, _modelo


def gerar_resposta(
    prompt: str
) -> str:
    """
    Gera uma resposta utilizando o modelo
    Qwen customizado com LoRA.
    """

    tokenizer, modelo = carregar_modelo()

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(modelo.device)

    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    tamanho_prompt = (
        inputs["input_ids"].shape[1]
    )

    resposta = tokenizer.decode(
        outputs[0][tamanho_prompt:],
        skip_special_tokens=True
    )

    resposta = resposta.strip()

    marcadores = [
        "\n### Observação:",
        "\n### Resposta:"
    ]

    for marcador in marcadores:
        if marcador in resposta:
            resposta = (
                resposta
                .split(marcador)[0]
                .strip()
            )

    return resposta