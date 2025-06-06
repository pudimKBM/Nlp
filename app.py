# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from transformers import pipeline

# Carregar modelos (pode levar algum tempo no primeiro uso, pois fará o download)
try:
    nlp_spacy = spacy.load("pt_core_news_lg") # Usando o modelo 'large' para melhor performance de NER
except OSError:
    print("Baixando o modelo pt_core_news_lg do spaCy...")
    spacy.cli.download("pt_core_news_lg")
    nlp_spacy = spacy.load("pt_core_news_lg")

# Carregar um modelo de análise de sentimento da Hugging Face
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
except Exception as e:
    print(f"Erro ao carregar modelo de sentimento: {e}. Verifique a conexão com a internet ou o nome do modelo.")
    sentiment_analyzer = None

app = FastAPI(
    title="API de Processamento e Resposta a Mensagens de Emergência",
    description="Processa mensagens, prioriza e gera respostas automáticas com base no contexto.",
    version="0.2.0",
)

# --- Modelo de Dados ---
class MensagemRequest(BaseModel):
    texto: str
    id_mensagem: str | None = None

class ResultadoProcessamento(BaseModel):
    id_mensagem: str | None = None
    texto_original: str
    palavras_chave: list[str]
    entidades_extraidas: dict[str, list[str]] # Ex: {"LOC": ["Avenida Paulista"], "PER": []}
    sentimento: dict
    score_prioridade: float
    prioridade: str
    sugestao_resposta: str

# --- Base de Conhecimento Local (Simulada) ---
# Em um sistema real, isso viria de um banco de dados ou outra API.
BASE_CONHECIMENTO_LOCAL = {
    "incêndio": {
        "medidas_seguranca": "Se possível, saia do local imediatamente. Não use elevadores. Cubra o nariz e a boca com um pano úmido para ajudar na respiração. Se não puder sair, vede as frestas das portas e sinalize sua presença em uma janela.",
        "contato_util": "Corpo de Bombeiros (193)"
    },
    "desabamento": {
        "medidas_seguranca": "Afaste-se da área de risco. Se estiver dentro de uma estrutura, procure abrigo sob uma viga ou móvel resistente e proteja a cabeça. Não retorne ao local até ser liberado pelas autoridades.",
        "contato_util": "Defesa Civil (199) e Bombeiros (193)"
    },
    "vazamento": { # Vazamento de gás
        "medidas_seguranca": "Não acione interruptores elétricos, não acenda fósforos ou isqueiros. Abra portas e janelas para ventilar o ambiente. Feche o registro de gás e saia do local.",
        "contato_util": "Distribuidora de gás da sua região e Bombeiros (193)."
    },
    "acidente": {
        "medidas_seguranca": "Sinalize o local para evitar novos acidentes. Verifique o estado das vítimas sem movê-las, a menos que haja risco iminente (fogo, explosão).",
        "contato_util": "SAMU (192) para vítimas, Polícia Militar (190) para trânsito."
    },
    "geral": { # Informações padrão
        "medidas_seguranca": "Mantenha a calma e aguarde as instruções das equipes de resgate. Forneça informações claras e precisas quando solicitado.",
        "contato_util": "Para emergências, ligue para 190 (Polícia), 192 (SAMU) ou 193 (Bombeiros)."
    }
}

# --- Palavras-Chave e Pontuação ---
PALAVRAS_CHAVE_EMERGENCIA = {
    "incêndio": 10, "fogo": 10, "preso": 8, "ferido": 9, "acidente": 8,
    "desabamento": 10, "afogamento": 9, "socorro": 7, "ajuda": 7, "urgente": 8,
    "sangue": 7, "dor": 6, "polícia": 7, "ambulância": 8, "bombeiro": 8,
    "desespero": 6, "perigo": 9, "criança": 7, "idoso": 7, "grávida": 7,
    "desmaiado": 8, "inconsciente": 9, "refém": 10, "sequestro": 10,
    "tiroteio": 9, "explosão": 10, "vazamento": 8, "contaminação": 8, "ameaça": 7
}
MODIFICADOR_SENTIMENTO = {"positive": -2, "neutral": 0, "negative": 5}

# --- Lógica do Pipeline NLP ---

def extrair_informacoes(texto: str) -> tuple[list[str], dict[str, list[str]]]:
    """Extrai palavras-chave e entidades nomeadas (NER) de um texto."""
    doc = nlp_spacy(texto.lower())
    # Extração de Palavras-chave
    palavras_chave = list(set([
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_ in PALAVRAS_CHAVE_EMERGENCIA
    ]))
    # Extração de Entidades (NER) - usando o texto original para maiúsculas/minúsculas
    doc_original = nlp_spacy(texto)
    entidades = {"LOC": [], "PER": [], "ORG": []}
    for ent in doc_original.ents:
        if ent.label_ in entidades:
            entidades[ent.label_].append(ent.text)
    return palavras_chave, entidades

def analisar_sentimento(texto: str) -> dict:
    """Analisa o sentimento do texto."""
    if sentiment_analyzer is None: return {"label": "INDISPONÍVEL", "score": 0.0}
    try:
        resultado = sentiment_analyzer(texto)[0]
        return {"label": resultado["label"].lower(), "score": round(resultado["score"], 4)}
    except Exception as e:
        print(f"Erro na análise de sentimento: {e}")
        return {"label": "ERRO", "score": 0.0}

def calcular_score_prioridade(palavras_chave: list[str], sentimento: dict) -> float:
    """Calcula um score de prioridade."""
    score = sum(PALAVRAS_CHAVE_EMERGENCIA.get(palavra, 0) for palavra in palavras_chave)
    score += MODIFICADOR_SENTIMENTO.get(sentimento.get("label", "neutral"), 0)
    return max(0, score)

def classificar_prioridade(score: float) -> str:
    if score >= 20: return "CRÍTICA"
    elif score >= 12: return "ALTA"
    elif score >= 7: return "MÉDIA"
    elif score > 0: return "BAIXA"
    else: return "INFORMATIVA"

# --- Gerador de Respostas ---

class GeradorDeRespostas:
    def __init__(self, base_conhecimento):
        self.kb = base_conhecimento
        self.templates = {
            "CRÍTICA": "ALERTA CRÍTICO: Emergência recebida sobre {evento} {localizacao}. Equipes de resgate enviadas com prioridade máxima. {medidas} Contato útil: {contato}.",
            "ALTA": "ALERTA DE ALTA PRIORIDADE: Recebemos sua mensagem sobre {evento} {localizacao}. Uma equipe está sendo designada para o local. {medidas} Contato útil: {contato}.",
            "MÉDIA": "Aviso recebido sobre {evento} {localizacao}. Sua solicitação está sendo processada. Siga estas orientações de segurança: {medidas} Em caso de necessidade, ligue para {contato}.",
            "BAIXA": "Obrigado por sua mensagem sobre {evento}. A situação foi registrada. Se a condição piorar, entre em contato novamente.",
            "INFORMATIVA": "Obrigado por sua mensagem. A informação foi registrada."
        }

    def gerar_resposta(self, prioridade: str, palavras_chave: list[str], entidades: dict) -> str:
        template = self.templates.get(prioridade, self.templates["INFORMATIVA"])
        if prioridade == "INFORMATIVA":
            return template

        # Identifica o evento principal e informações de segurança
        evento_principal = "uma situação de emergência"
        info_kb = self.kb["geral"] # Padrão
        for palavra in palavras_chave:
            if palavra in self.kb:
                evento_principal = palavra
                info_kb = self.kb[palavra]
                break

        # Adiciona localização se disponível
        locais = entidades.get("LOC", [])
        localizacao_str = f"em '{', '.join(locais)}'" if locais else "na sua área"

        return template.format(
            evento=evento_principal,
            localizacao=localizacao_str,
            medidas=info_kb.get("medidas_seguranca", ""),
            contato=info_kb.get("contato_util", "")
        ).strip()

# Inicializa o gerador de respostas
response_generator = GeradorDeRespostas(BASE_CONHECIMENTO_LOCAL)

# --- Endpoint da API ---

@app.post("/processar_mensagem/", response_model=ResultadoProcessamento)
async def processar_mensagem_endpoint(request: MensagemRequest):
    texto = request.texto
    if not texto.strip():
        raise HTTPException(status_code=400, detail="O texto da mensagem não pode estar vazio.")

    # Pipeline NLP
    palavras_chave, entidades = extrair_informacoes(texto)
    sentimento = analisar_sentimento(texto)
    score = calcular_score_prioridade(palavras_chave, sentimento)
    prioridade = classificar_prioridade(score)
    sugestao_resposta = response_generator.gerar_resposta(prioridade, palavras_chave, entidades)

    return ResultadoProcessamento(
        id_mensagem=request.id_mensagem,
        texto_original=texto,
        palavras_chave=palavras_chave,
        entidades_extraidas=entidades,
        sentimento=sentimento,
        score_prioridade=score,
        prioridade=prioridade,
        sugestao_resposta=sugestao_resposta
    )

# --- Instruções de Execução ---
# (As mesmas de antes, mas lembre-se de instalar 'pt_core_news_lg' se necessário)
# 1. pip install fastapi uvicorn "spacy>=3.0" transformers torch sentencepiece
# 2. python -m spacy download pt_core_news_lg
# 3. uvicorn main:app --reload
#
# Exemplo de teste com localização:
# {
#   "texto": "SOCORRO! Há um grande incêndio no Edifício Copan na Avenida Ipiranga. Estou preso no 10º andar, muita fumaça!",
#   "id_mensagem": "MSG456"
# }
#
# Resposta esperada:
# {
#   ...
#   "prioridade": "CRÍTICA",
#   "entidades_extraidas": {
#     "LOC": ["Edifício Copan", "Avenida Ipiranga"], ...
#   },
#   "sugestao_resposta": "ALERTA CRÍTICO: Emergência recebida sobre incêndio em 'Edifício Copan, Avenida Ipiranga'. Equipes de resgate enviadas com prioridade máxima. Se possível, saia do local imediatamente. Não use elevadores... Contato útil: Corpo de Bombeiros (193)."
# }
