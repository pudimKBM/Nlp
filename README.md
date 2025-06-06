# API de Processamento e Resposta a Mensagens de Emergência

## Integrantes
* Antônio Libarato RM558014
* Renan de França Gonçalves RM558413
* Thiago Almança RM558108



## Visão Geral

Este projeto implementa uma API para processar mensagens de emergência. A API analisa o texto da mensagem para extrair informações chave, determinar o sentimento, priorizar a mensagem e gerar uma resposta automática apropriada. O sistema é projetado para auxiliar no gerenciamento e resposta a situações de emergência de forma mais eficiente, utilizando técnicas de Processamento de Linguagem Natural (PLN).

## Funcionalidades

*   **Processamento de Linguagem Natural (PLN):** Utiliza a biblioteca spaCy para Reconhecimento de Entidades Nomeadas (NER) e extração de palavras-chave relevantes.
*   **Análise de Sentimento:** Emprega um modelo da Hugging Face Transformers para analisar o sentimento expresso na mensagem (positivo, neutro, negativo).
*   **Priorização Inteligente:** Calcula um score de prioridade com base nas palavras-chave identificadas e no sentimento da mensagem.
*   **Geração de Resposta Automática:** Cria respostas contextuais com base na prioridade da mensagem, no tipo de emergência detectado e nas informações extraídas (como localização).
*   **API RESTful:** Fornece um endpoint (`/processar_mensagem/`) para o processamento de mensagens, construído com FastAPI.

## Funcionamento do Pipeline de PLN

O pipeline de processamento de mensagens segue as seguintes etapas:

1.  **Recepção da Mensagem:** A API recebe uma mensagem de texto através de uma requisição POST no endpoint `/processar_mensagem/`.
2.  **Extração de Informações (spaCy):**
    *   O texto da mensagem é processado pelo modelo `pt_core_news_lg` do spaCy.
    *   **Palavras-chave:** São extraídas palavras-chave relevantes para emergências (ex: "incêndio", "ferido", "socorro") após lematização e remoção de stop words e pontuações.
    *   **Entidades Nomeadas (NER):** São identificadas entidades como Localizações (LOC), Pessoas (PER) e Organizações (ORG) no texto original.
3.  **Análise de Sentimento (Hugging Face Transformers):**
    *   O sentimento da mensagem é analisado utilizando o modelo pré-treinado `lxyuan/distilbert-base-multilingual-cased-sentiments-student`. O resultado inclui um rótulo (ex: "negative") e um score de confiança.
4.  **Cálculo do Score de Prioridade:**
    *   Um score numérico é calculado somando pesos associados às palavras-chave de emergência encontradas.
    *   Um modificador é aplicado com base no sentimento detectado (ex: sentimento negativo aumenta o score).
    *   O score final é usado para classificar a prioridade da mensagem (CRÍTICA, ALTA, MÉDIA, BAIXA, INFORMATIVA).
5.  **Geração de Resposta:**
    *   Com base na prioridade, palavras-chave e entidades, uma sugestão de resposta é gerada.
    *   O sistema utiliza uma base de conhecimento local (`BASE_CONHECIMENTO_LOCAL`) que contém medidas de segurança e contatos úteis para diferentes tipos de emergência (ex: incêndio, desabamento).
    *   Templates de resposta são preenchidos dinamicamente com o evento principal, localização (se disponível), medidas de segurança e contatos.
6.  **Retorno:** A API retorna um objeto JSON contendo o ID da mensagem (se fornecido), o texto original, as palavras-chave extraídas, as entidades, o sentimento, o score de prioridade, o nível de prioridade e a sugestão de resposta.

## Tecnologias Utilizadas

*   Python 3.x
*   FastAPI: Para a construção da API web.
*   Uvicorn: Servidor ASGI para rodar a aplicação FastAPI.
*   spaCy (`pt_core_news_lg`): Para tarefas de PLN como NER e tokenização.
*   Transformers (Hugging Face): Para análise de sentimento com o modelo `lxyuan/distilbert-base-multilingual-cased-sentiments-student`.
*   PyTorch: Backend para a biblioteca Transformers.
*   Pydantic: Para validação de dados e gerenciamento de configurações no FastAPI.

## Configuração e Execução

### Pré-requisitos

*   Python 3.8 ou superior
*   pip (gerenciador de pacotes Python)

### Instalação

1.  **Clone o repositório (se aplicável):**
    ```bash
    git clone <url-do-seu-repositorio>
    cd <nome-do-repositorio>
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    ```
    No Windows:
    ```bash
    venv\Scripts\activate
    ```
    No macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixe o modelo de linguagem do spaCy para português:**
    ```bash
    python -m spacy download pt_core_news_lg
    ```
    *Observação: A aplicação tentará baixar este modelo na primeira execução caso não o encontre, mas o download manual é recomendado para garantir.*

### Executando a API

Para iniciar o servidor da API, execute o seguinte comando no terminal, a partir do diretório raiz do projeto:

```bash
uvicorn app:app --reload
```


```bash
curl -X POST "http://127.0.0.1:8000/processar_mensagem/" \
-H "Content-Type: application/json" \
-d '{
  "texto": "SOCORRO! Há um grande incêndio no Edifício Copan na Avenida Ipiranga. Estou preso no 10º andar, muita fumaça!",
  "id_mensagem": "MSG456"
}'

```


# Demonstração Funcional
## A funcionalidade da API pode ser demonstrada da seguinte forma:

### Vídeo de Demonstração

Assista a uma demonstração da API em funcionamento:
https://youtu.be/e0DskLuqUus

* Inicie o servidor da API conforme descrito na seção "Executando a API".
* Envie requisições POST para o endpoint /processar_mensagem/ utilizando:
* Ferramentas de linha de comando como curl (veja exemplo acima).
* Clientes API gráficos como Postman ou Insomnia.
* A documentação interativa da API (Swagger UI) fornecida pelo FastAPI, acessível em http://127.0.0.1:8000/docs quando o * servidor está em execução.
* Observe a resposta JSON retornada pela API, que incluirá as informações processadas, a prioridade e a sugestão de resposta.
* Exemplos de Mensagens para Demonstração:
* Prioridade CRÍTICA (Incêndio com localização detalhada):

### Input:
```json
{
  "texto": "Fogo no prédio da Rua Augusta, 123! Muitas chamas e fumaça saindo das janelas do terceiro andar. Pessoas gritando por ajuda.",
  "id_mensagem": "MSG001"
}
```
* Resultado Esperado: Prioridade "CRÍTICA", palavras-chave como "fogo", "chamas", "ajuda", entidade "LOC" com "Rua Augusta, 123", e uma resposta específica para incêndio com instruções de segurança e contato dos bombeiros.
Prioridade ALTA (Acidente com feridos):

### Input:
```json
{
  "texto": "Acho que vi um acidente de carro na Marginal Tietê perto da ponte. Parece que tem gente ferida.",
  "id_mensagem": "MSG002"
}
```
* Resultado Esperado: Prioridade "ALTA" ou "MÉDIA", palavras-chave como "acidente", "ferida", entidade "LOC" com "Marginal Tietê", e uma resposta para acidentes com contato do SAMU e Polícia.
Prioridade BAIXA/INFORMATIVA (Consulta):

### Input:
```json
{
  "texto": "Qual o telefone da defesa civil para informações sobre alagamentos?",
  "id_mensagem": "MSG003"
}
```
* Resultado Esperado: Prioridade "BAIXA" ou "INFORMATIVA", poucas palavras-chave de emergência, e uma resposta mais genérica ou informativa, possivelmente indicando o contato solicitado se estiver na base de conhecimento.
Mensagem com Forte Sentimento Negativo:

### Input:
```json
{
  "texto": "Estou desesperado, meu vizinho está passando muito mal e não consigo chamar uma ambulância! Ele precisa de socorro urgente!",
  "id_mensagem": "MSG004"
}
```
* Resultado Esperado: Score de prioridade elevado devido ao sentimento "negative" e palavras-chave como "desesperado", "ambulância", "socorro", "urgente", resultando em prioridade "ALTA" ou "CRÍTICA".
