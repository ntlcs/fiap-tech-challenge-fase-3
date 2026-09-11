# Tech Challenge Fase 3 — Assistente Clínico com LLM

Projeto desenvolvido para o **Tech Challenge — Fase 3 da Pós Tech FIAP em Inteligência Artificial para Devs**.

O projeto implementa uma prova de conceito acadêmica de um assistente virtual de apoio clínico para **Diabetes Mellitus Tipo 2 (DM2)**, utilizando fine-tuning de LLM, RAG, LangChain, LangGraph, base de dados estruturada e mecanismos de segurança e auditoria.

> **Aviso:** este projeto possui finalidade exclusivamente acadêmica. O assistente não substitui profissionais de saúde, não realiza diagnóstico definitivo e não deve prescrever medicamentos ou doses. As respostas geradas exigem validação humana.

---

## Objetivo

Construir um assistente capaz de:

- utilizar uma LLM customizada com dados médicos sintéticos;
- consultar informações estruturadas de pacientes;
- recuperar protocolos institucionais relevantes;
- contextualizar respostas com dados do paciente;
- coordenar o fluxo de decisão com LangGraph;
- bloquear solicitações ou respostas potencialmente inadequadas;
- apresentar as fontes utilizadas;
- exigir validação médica;
- registrar as execuções para auditoria.

---

## Arquitetura

O fluxo principal da solução é:

```text
Pergunta + patient_id
        |
        v
Validação de entrada
        |
        +---- solicitação proibida ----> Bloqueio
        |
        v
Consulta ao SQLite
        |
        +---- paciente inexistente ----> Resposta segura
        |
        v
Recuperação de protocolos (RAG + FAISS)
        |
        v
LangChain
        |
        v
LLM Qwen 2.5 + LoRA
        |
        v
Validação da saída
        |
        +---- saída insegura ----------> Bloqueio
        |
        v
Resposta contextualizada
        |
        v
Fontes + validação humana
        |
        v
Logging / Auditoria
```

A orquestração do fluxo é realizada com **LangGraph**.

---

## Tecnologias utilizadas

- Python
- Google Colab
- Qwen2.5-0.5B-Instruct
- Hugging Face Transformers
- PEFT / LoRA
- LangChain
- LangGraph
- Sentence Transformers
- FAISS
- SQLite
- Pandas
- PyTorch

---

## Fine-tuning

Foi utilizado como modelo base:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

O ajuste fino foi realizado com **LoRA (Low-Rank Adaptation)**.

Principais parâmetros:

```text
r = 8
lora_alpha = 16
lora_dropout = 0.05
epochs = 3
learning_rate = 2e-4
```

Foram treinados aproximadamente **1,08 milhão de parâmetros**, correspondendo a cerca de **0,22% dos parâmetros do modelo**.

O treinamento foi realizado em ambiente Google Colab.

### Resultados do treinamento

Ao longo das três épocas, a perda de validação apresentou redução:

```text
Época 1: eval_loss = 2.122232
Época 2: eval_loss = 1.920492
Época 3: eval_loss = 1.848623
```

A acurácia por token evoluiu aproximadamente de:

```text
0.6021 → 0.6386 → 0.6470
```

Esses resultados são utilizados apenas como indicadores experimentais da prova de conceito e não representam validação clínica do modelo.

---

## Dados

Foram utilizados dados sintéticos relacionados ao acompanhamento de pacientes com Diabetes Mellitus Tipo 2.

O preprocessing contempla:

- anonimização;
- remoção de informações pessoalmente identificáveis (PII);
- normalização;
- tratamento de valores ausentes;
- remoção de duplicidades;
- validação de faixas plausíveis;
- curadoria dos registros.

O dataset final utilizado para fine-tuning contém **58 exemplos**, separados em:

```text
46 exemplos de treinamento
12 exemplos de validação
```

O tamanho reduzido é adequado apenas à demonstração acadêmica da arquitetura e não ao uso clínico real.

---

## Base estruturada

Foi criada uma base SQLite contendo pacientes sintéticos e informações como:

- idade;
- sexo;
- diagnóstico;
- glicemia;
- HbA1c;
- pressão arterial;
- IMC;
- creatinina;
- colesterol;
- exames pendentes.

O assistente consulta essa base utilizando o `patient_id`.

---

## RAG e protocolos institucionais

Os protocolos sintéticos são indexados utilizando:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Os embeddings são armazenados em um índice **FAISS**.

Foram utilizados protocolos relacionados a:

- controle glicêmico;
- avaliação renal;
- avaliação oftalmológica;
- segurança do assistente clínico.

As fontes recuperadas pelo RAG são incluídas na resposta para aumentar a rastreabilidade e explainability.

---

## LangChain

O LangChain é responsável pela composição do pipeline que conecta:

```text
Pergunta
+
Dados estruturados do paciente
+
Protocolos recuperados pelo RAG
+
LLM customizada
```

O prompt determina que o modelo utilize somente as informações fornecidas no contexto e estabelece restrições de segurança.

---

## LangGraph

O LangGraph coordena o fluxo clínico:

1. validação da entrada;
2. consulta do paciente;
3. verificação da existência do paciente;
4. recuperação de protocolos;
5. geração da resposta;
6. validação da saída;
7. resposta segura ou bloqueio;
8. auditoria.

Isso permite separar as responsabilidades e aplicar regras determinísticas antes e depois da geração da LLM.

---

## Segurança

O assistente possui guardrails para bloquear solicitações relacionadas, entre outros casos, a:

- prescrição direta;
- indicação de dose.

A saída da LLM também passa por validação para detectar padrões considerados inadequados.

Toda resposta clínica mantém:

```text
human_validation_required = True
```

Portanto, a solução atua exclusivamente como ferramenta de **apoio à decisão**, e não como substituta da decisão médica.

---

## Logging e auditoria

Cada execução registra informações de auditoria em formato JSONL, incluindo:

- identificador da auditoria;
- timestamp;
- patient_id;
- pergunta;
- resultado;
- bloqueios;
- motivos de segurança;
- fontes utilizadas;
- necessidade de validação humana.

Um exemplo está disponível em:

```text
logs/clinical_assistant.jsonl
```

Os dados utilizados neste projeto são sintéticos. Em um ambiente hospitalar real seriam necessários controles adicionais de segurança, acesso, criptografia, retenção e proteção de dados sensíveis.

---

## Validação final

A arquitetura foi submetida a seis cenários controlados:

1. consulta clínica geral;
2. acompanhamento renal;
3. acompanhamento oftalmológico;
4. solicitação de prescrição;
5. solicitação de dose;
6. paciente inexistente.

Resultado:

```text
Testes aprovados: 6/6
Taxa de sucesso arquitetural: 100%
Bloqueios na entrada: 2
Bloqueios na saída da LLM: 0
Execuções auditadas: 6/6
Validação humana obrigatória: 6/6
Tempo médio por cenário: 37.896 segundos
```

A taxa de 100% refere-se exclusivamente ao atendimento dos critérios funcionais definidos para os seis cenários controlados. **Não representa acurácia clínica, eficácia médica ou garantia absoluta de segurança.**

Os resultados estão disponíveis em:

```text
docs/report/validation_results/
```

---

## Estrutura do projeto

```text
fiap-tech-challenge-fase-3/
|
├── data/
│   ├── database/
│   ├── processed/
│   └── synthetic/
|
├── docs/
│   └── report/
|
├── logs/
|
├── models/
│   └── qwen2.5_0.5b_lora/
|
├── src/
│   ├── __init__.py
│   ├── assistant.py
│   ├── config.py
│   ├── database.py
│   ├── graph.py
│   ├── llm.py
│   ├── logger.py
│   ├── preprocessing.py
│   ├── rag.py
│   └── safety.py
|
├── 03_Desafio_FIAP_IA_01_preprocessing.ipynb
├── 03_Desafio_FIAP_IA_02_fine_tuning.ipynb
├── 03_Desafio_FIAP_IA_03_evaluation.ipynb
├── 03_Desafio_FIAP_IA_04_database.ipynb
├── 03_Desafio_FIAP_IA_05_rag.ipynb
├── 03_Desafio_FIAP_IA_06_assistant_langchain.ipynb
├── 03_Desafio_FIAP_IA_07_safety_langgraph.ipynb
└── 03_Desafio_FIAP_IA_08_final_validation.ipynb
```

---

## Notebooks

| Notebook | Finalidade |
|---|---|
| 01 | Preprocessing, anonimização e curadoria |
| 02 | Fine-tuning da LLM com LoRA |
| 03 | Avaliação experimental do modelo |
| 04 | Base estruturada SQLite |
| 05 | RAG e indexação FAISS |
| 06 | Assistente clínico com LangChain |
| 07 | Segurança, LangGraph e auditoria |
| 08 | Validação integrada da arquitetura |

Os notebooks foram desenvolvidos e executados no **Google Colab**.

---

## Execução

A execução de referência do projeto foi realizada no Google Colab.

A sequência recomendada é:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

Os notebooks instalam/importam as dependências necessárias para cada etapa e utilizam os artefatos produzidos nas etapas anteriores.

Os caminhos utilizados durante a execução no Colab apontam para a estrutura:

```text
/content/drive/MyDrive/FIAP/TechChallenge_Fase3/
```

Para execução em outro ambiente, os caminhos devem ser ajustados conforme o local onde o projeto estiver armazenado.

---

## Limitações

Esta solução é uma **prova de conceito acadêmica**.

Entre as principais limitações estão:

- quantidade reduzida de exemplos de fine-tuning;
- pacientes e protocolos sintéticos;
- avaliação em conjunto pequeno de cenários;
- possibilidade de alucinação da LLM;
- guardrails baseados parcialmente em regras;
- ausência de validação clínica prospectiva;
- necessidade obrigatória de supervisão humana.

O sistema não deve ser utilizado para tomada autônoma de decisão médica.

---

## Autoria

Projeto desenvolvido para o **Tech Challenge — Fase 3 da Pós Tech FIAP — Inteligência Artificial para Devs**.
