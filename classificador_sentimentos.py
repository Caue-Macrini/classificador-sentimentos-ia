# -*- coding: utf-8 -*-
"""
============================================================================
 CLASSIFICADOR DE SENTIMENTOS  -  Naive Bayes do zero (Python puro)
============================================================================

INTUICAO (o "porque" e o "como" em nivel conceitual)
----------------------------------------------------------------------------
Queremos que o computador leia uma frase e diga se ela e POSITIVA ou NEGATIVA.

A ideia central: certas palavras aparecem muito mais em frases positivas
("adorei", "otimo", "excelente") e outras em frases negativas ("odiei",
"pessimo", "horrivel"). Se a gente contar essas frequencias, da pra estimar
a probabilidade de uma frase nova ser positiva ou negativa.

Usamos o Teorema de Bayes. Para uma frase, calculamos:

    P(classe | frase)  ~  P(classe) * PRODUTO de P(palavra | classe)

E escolhemos a classe (positivo/negativo) com maior probabilidade.
O "Naive" (ingenuo) vem de assumirmos que as palavras sao independentes
entre si - uma simplificacao que, na pratica, funciona muito bem para texto.

Dois detalhes importantes de engenharia:
  1) SUAVIZACAO DE LAPLACE: somamos 1 na contagem de cada palavra pra nunca
     ter probabilidade zero (uma palavra nova nao "zera" a frase inteira).
  2) LOGARITMOS: multiplicar muitas probabilidades pequenas gera numeros
     minusculos (underflow). Somar logaritmos e equivalente e estavel.

BONUS - TRATAMENTO DE NEGACAO:
  "gostei" e positivo, mas "nao gostei" e negativo. Marcamos as palavras
  logo apos uma negacao ("nao", "nunca"...) com o prefixo "NAO_", para que
  "NAO_gostei" seja aprendido separadamente. Simples e eficaz.
============================================================================
"""

import re
import math
from collections import defaultdict, Counter

from dados_treino import DADOS_TREINO, DADOS_TESTE


# Palavras que invertem o sentido da(s) palavra(s) seguinte(s).
NEGACOES = {"nao", "nunca", "nem", "jamais", "sem", "nenhum", "nenhuma"}

# Palavras muito comuns que carregam pouco sentimento (stopwords).
# Remove-las reduz ruido. Mantemos negacoes de fora dessa lista de proposito.
STOPWORDS = {
    "a", "o", "e", "de", "do", "da", "que", "com", "em", "um", "uma",
    "os", "as", "no", "na", "para", "por", "eu", "me", "se", "muito",
    "mais", "ao", "meu", "minha", "esse", "essa", "isso", "ja", "the",
}


def remover_acentos(texto):
    """Normaliza acentos de forma simples (a=a, e=e...). Deixa o vocabulario
    mais consistente: 'otimo' e 'ótimo' viram a mesma palavra."""
    mapa = str.maketrans("áàâãäéèêëíìîïóòôõöúùûüç", "aaaaaeeeeiiiiooooouuuuc")
    return texto.translate(mapa)


def preprocessar(texto):
    """Transforma uma frase crua numa lista de "tokens" (palavras limpas).

    Passos:
      1. minusculas + remover acentos
      2. manter so letras (tira pontuacao, numeros, emojis)
      3. tratar negacao: apos uma negacao, prefixar as 2 palavras seguintes
      4. remover stopwords
    """
    texto = remover_acentos(texto.lower())
    # \w com re.UNICODE ainda pegaria acentos; ja removemos, entao [a-z] basta
    palavras = re.findall(r"[a-z]+", texto)

    tokens = []
    negar_contador = 0  # quantas proximas palavras devem receber prefixo NAO_
    for palavra in palavras:
        if palavra in NEGACOES:
            negar_contador = 2          # nega as proximas 2 palavras
            continue                    # a propria negacao nao vira token
        if palavra in STOPWORDS:
            continue
        if negar_contador > 0:
            tokens.append("NAO_" + palavra)
            negar_contador -= 1
        else:
            tokens.append(palavra)
    return tokens


class ClassificadorNaiveBayes:
    """Classificador de sentimentos Multinomial Naive Bayes."""

    def __init__(self):
        self.classes = []                       # ex: ["positivo", "negativo"]
        self.log_prior = {}                     # log P(classe)
        self.contagem_palavras = {}             # classe -> Counter de palavras
        self.total_palavras_classe = {}         # classe -> total de tokens
        self.vocabulario = set()                # todas as palavras conhecidas

    def treinar(self, dados):
        """'Aprende' a partir dos exemplos (texto, rotulo)."""
        docs_por_classe = defaultdict(int)
        self.contagem_palavras = defaultdict(Counter)

        for texto, classe in dados:
            docs_por_classe[classe] += 1
            for token in preprocessar(texto):
                self.contagem_palavras[classe][token] += 1
                self.vocabulario.add(token)

        self.classes = list(docs_por_classe.keys())
        total_docs = sum(docs_por_classe.values())

        for classe in self.classes:
            # P(classe) = fracao de documentos daquela classe
            self.log_prior[classe] = math.log(docs_por_classe[classe] / total_docs)
            self.total_palavras_classe[classe] = sum(
                self.contagem_palavras[classe].values()
            )

    def _log_prob_palavra(self, palavra, classe):
        """log P(palavra | classe) com suavizacao de Laplace (+1)."""
        contagem = self.contagem_palavras[classe][palavra]
        numerador = contagem + 1
        denominador = self.total_palavras_classe[classe] + len(self.vocabulario)
        return math.log(numerador / denominador)

    def pontuar(self, texto):
        """Retorna um dicionario classe -> log-probabilidade da frase."""
        tokens = preprocessar(texto)
        pontuacoes = {}
        for classe in self.classes:
            score = self.log_prior[classe]
            for token in tokens:
                # ignora palavras que nunca vimos em nenhum treino
                if token in self.vocabulario:
                    score += self._log_prob_palavra(token, classe)
            pontuacoes[classe] = score
        return pontuacoes

    def prever(self, texto):
        """Retorna (classe_prevista, confianca_percentual)."""
        pontuacoes = self.pontuar(texto)
        classe_prevista = max(pontuacoes, key=pontuacoes.get)
        confianca = self._softmax_confianca(pontuacoes, classe_prevista)
        return classe_prevista, confianca

    @staticmethod
    def _softmax_confianca(pontuacoes, classe):
        """Converte log-probabilidades em % (0-100) para exibir a 'confianca'."""
        maximo = max(pontuacoes.values())
        exps = {c: math.exp(s - maximo) for c, s in pontuacoes.items()}
        soma = sum(exps.values())
        return 100.0 * exps[classe] / soma

    def palavras_decisivas(self, texto, n=3):
        """Explica a decisao: quais palavras da frase mais 'puxaram' o
        resultado para a classe vencedora (diferenca de log-prob entre
        as duas classes). Otimo para a apresentacao!"""
        if len(self.classes) != 2:
            return []
        tokens = [t for t in preprocessar(texto) if t in self.vocabulario]
        c1, c2 = self.classes
        contribs = []
        for token in set(tokens):
            diff = self._log_prob_palavra(token, c1) - self._log_prob_palavra(token, c2)
            # a favor de c1 se diff > 0, a favor de c2 se diff < 0
            favor = c1 if diff > 0 else c2
            contribs.append((token, favor, abs(diff)))
        contribs.sort(key=lambda x: x[2], reverse=True)
        return contribs[:n]


def avaliar(modelo, dados_teste):
    """Mede a acuracia: fracao de frases de teste classificadas corretamente."""
    acertos = 0
    for texto, esperado in dados_teste:
        previsto, _ = modelo.prever(texto)
        if previsto == esperado:
            acertos += 1
    return acertos / len(dados_teste)


def barra(percentual, largura=20):
    """Desenha uma barrinha de progresso em texto: [#######----] """
    cheios = int(round(percentual / 100 * largura))
    return "[" + "#" * cheios + "-" * (largura - cheios) + "]"


def analisar_frase(modelo, frase):
    """Imprime a analise completa e didatica de uma frase."""
    classe, confianca = modelo.prever(frase)
    carinha = ":)" if classe == "positivo" else ":("
    print(f"\n  Frase:      \"{frase}\"")
    print(f"  Tokens:     {preprocessar(frase)}")
    print(f"  Resultado:  {carinha}  {classe.upper()}  ({confianca:.1f}% de confianca)")
    print(f"              {barra(confianca)}")
    decisivas = modelo.palavras_decisivas(frase)
    if decisivas:
        print("  Por que?    Palavras que mais pesaram:")
        for palavra, favor, peso in decisivas:
            print(f"                - '{palavra}'  ->  puxa para {favor}")


def modo_interativo(modelo):
    """Loop para a turma digitar frases e ver a IA classificando ao vivo."""
    print("\n" + "=" * 68)
    print("  MODO INTERATIVO - digite uma frase e a IA diz o sentimento")
    print("  (digite 'sair' para encerrar)")
    print("=" * 68)
    while True:
        try:
            frase = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAte mais!")
            break
        if frase.lower() in {"sair", "exit", "quit", ""}:
            print("Ate mais!")
            break
        analisar_frase(modelo, frase)


def main():
    print("=" * 68)
    print("  CLASSIFICADOR DE SENTIMENTOS  (Naive Bayes - Python puro)")
    print("=" * 68)

    # 1) TREINAR: a IA "aprende" com os exemplos rotulados
    modelo = ClassificadorNaiveBayes()
    modelo.treinar(DADOS_TREINO)
    print(f"\n  Treinado com {len(DADOS_TREINO)} exemplos.")
    print(f"  Vocabulario aprendido: {len(modelo.vocabulario)} palavras distintas.")

    # 2) AVALIAR: quao bem ela acerta em frases que nunca viu?
    acuracia = avaliar(modelo, DADOS_TESTE)
    print(f"  Acuracia no conjunto de teste: {acuracia * 100:.1f}% "
          f"({int(acuracia * len(DADOS_TESTE))}/{len(DADOS_TESTE)} corretas)")

    # 3) DEMONSTRAR: alguns exemplos prontos
    print("\n" + "-" * 68)
    print("  EXEMPLOS DE DEMONSTRACAO")
    print("-" * 68)
    exemplos = [
        "Adorei esse produto, recomendo muito!",
        "Que experiencia horrivel, nunca mais compro.",
        "Nao gostei, esperava muito mais.",         # negacao invertendo!
        "O atendimento foi excelente e super rapido.",
    ]
    for frase in exemplos:
        analisar_frase(modelo, frase)

    # 4) INTERAGIR
    modo_interativo(modelo)


if __name__ == "__main__":
    main()
