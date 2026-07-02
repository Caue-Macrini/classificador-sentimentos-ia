# Classificador de Sentimentos com IA (Naive Bayes)

Protótipo de **Processamento de Linguagem Natural (PLN)** que lê uma frase em
português e decide se o sentimento é **positivo** ou **negativo**.
Implementado em **Python puro** (só a biblioteca padrão — não precisa instalar nada).

---

## Como rodar

```bash
cd trabalho_ia_sentimentos
python classificador_sentimentos.py
```

O programa:
1. **Treina** a IA com 60 exemplos rotulados.
2. **Mostra a acurácia** em um conjunto de teste (frases que ela nunca viu).
3. **Demonstra** alguns exemplos prontos.
4. Abre o **modo interativo**: digite qualquer frase e veja a IA classificar ao vivo.
   Digite `sair` para encerrar.

---

## Arquivos

| Arquivo | O que é |
|---|---|
| `dados_treino.py` | A "experiência" da IA: frases rotuladas como positivas/negativas |
| `classificador_sentimentos.py` | O algoritmo Naive Bayes + interface interativa |

---

# Roteiro da apresentação

## 1. Intuição (o "porquê" e o "como")

**Problema:** dado um texto (ex.: avaliação de um produto), descobrir
automaticamente se a opinião é boa ou ruim. Útil para empresas analisarem
milhares de comentários sem ler um por um.

**Ideia central:** certas palavras aparecem muito mais em frases positivas
(*adorei, ótimo, excelente*) do que em negativas (*odiei, péssimo, horrível*).
Se contarmos essas frequências no treino, conseguimos estimar a probabilidade
de uma frase nova ser positiva ou negativa.

**O algoritmo (Naive Bayes):** usa o **Teorema de Bayes**. Para uma frase,
calculamos de forma simplificada:

```
P(classe | frase)  ∝  P(classe) × Π P(palavra | classe)
```

e escolhemos a classe com maior probabilidade. O nome "Naive" (ingênuo) vem de
assumirmos que as palavras são **independentes** entre si — uma simplificação
que funciona surpreendentemente bem para texto.

**Dois truques de engenharia** (bons de citar):
- **Suavização de Laplace (+1):** evita que uma palavra nunca vista "zere" a
  frase inteira.
- **Logaritmos:** somar logs em vez de multiplicar probabilidades minúsculas
  evita erro numérico (*underflow*).

**Diferencial — tratamento de negação:** *"gostei"* é positivo, mas *"não
gostei"* é negativo. O código marca as palavras após uma negação com o prefixo
`NAO_`, então `NAO_gostei` é aprendido como algo negativo. (Mostre isso na demo!)

## 2. Código (trechos-chave)

- **`preprocessar()`** — limpa o texto: minúsculas, remove acentos e pontuação,
  aplica o tratamento de negação e remove *stopwords* (palavras sem sentimento).
- **`treinar()`** — percorre os exemplos contando quantas vezes cada palavra
  aparece em cada classe. É aqui que a IA "aprende".
- **`_log_prob_palavra()`** — calcula `log P(palavra | classe)` com Laplace.
- **`prever()`** — soma os logs e escolhe a classe vencedora + a confiança (%).
- **`palavras_decisivas()`** — explica **por que** a IA decidiu, mostrando as
  palavras que mais pesaram. Ótimo para deixar a "caixa-preta" transparente.

**Bibliotecas usadas:** apenas padrão do Python — `re` (limpeza de texto com
expressões regulares), `math` (logaritmos) e `collections` (contadores).

## 3. Funcionamento (demonstração)

Sugestão de frases para testar ao vivo com a turma:

| Frase | Esperado |
|---|---|
| `adorei, produto excelente e entrega rápida` | Positivo |
| `que experiência horrível, nunca mais compro` | Negativo |
| `não gostei, esperava muito mais` | Negativo (negação!) |
| `atendimento maravilhoso, recomendo demais` | Positivo |

**Limitações (mostre honestidade técnica):**
- Aprende só o vocabulário do treino — palavras totalmente novas são ignoradas.
- Não entende **sarcasmo** nem contexto complexo ("nossa, que ótimo..." irônico).
- Base de treino pequena (60 frases); mais dados = mais robusto.
- Ignora a **ordem** das palavras (é o preço da simplificação "naive").

Essas limitações são o gancho perfeito para dizer como se evoluiria o projeto:
mais dados, modelos maiores (redes neurais / *embeddings*), etc.
