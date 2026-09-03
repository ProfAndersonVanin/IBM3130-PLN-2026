# IBM3130 — Processamento de Linguagem Natural
## Aula 04 · Semana 4 · Representações de Texto — Bag of Words e TF-IDF

---

| | |
|---|---|
| **Disciplina** | IBM3130 — Processamento de Linguagem Natural |
| **Aula** | 04 de 14 · Semana 4 · 2º Semestre de 2026 |
| **Módulo** | Módulo 2 — Representações de Texto (início) |
| **Pré-requisito** | Aulas 01–03 — PLN, Pré-processamento, POS Tagging |
| **Ferramentas** | Google Colab · Python · NLTK · spaCy |
| **Referências** | B1 Cap. 6 (pp. 149–180) · B2 Cap. 5 e 8 · C3 Cap. 4 |

---

## Contextualização

Nas aulas anteriores aprendemos a preparar textos com NLTK e spaCy:
tokenizar, remover stopwords, aplicar stemming e identificar classes gramaticais.

Mas algoritmos de Machine Learning trabalham apenas com **números**, não com palavras.
Precisamos transformar cada texto em uma lista de números — um **vetor**.

Essa transformação se chama **vetorização de texto** e é o que estudamos nesta aula.

### O problema que precisamos resolver

```
Review A: 'produto excelente, chegou rápido, adorei!'
Review B: 'produto horrível, chegou quebrado, odeiei!'
```

Ambas contêm `'produto'` e `'chegou'` — mas têm sentimentos opostos.
Uma boa representação numérica precisa capturar as palavras que **diferenciam** as duas.

### Os dois métodos desta aula

| Método | O que faz | Limitação |
|--------|-----------|-----------|
| **Bag of Words** | Conta quantas vezes cada palavra aparece | Palavras comuns dominam |
| **TF-IDF** | Dá peso maior a palavras raras e específicas | Ainda ignora a ordem |

---

# PARTE 1 — Do Texto ao Número

## 1.1 Por que vetores?

Algoritmos de ML precisam de números. A representação vetorial transforma
cada documento em um vetor `x ∈ ℝᵏ` onde `k` é o tamanho do vocabulário.

Documentos similares devem ter vetores **próximos** no espaço vetorial.

## 1.2 O que toda representação precisa responder

```
1. Quais são as dimensões do espaço? → o vocabulário
2. Qual valor atribuir a cada dimensão? → a função de peso
3. Como tratar documentos de tamanhos diferentes? → normalização
4. Como capturar significado e não apenas forma? → semântica (Aula 06)
```

---

# PARTE 2 — Bag of Words

## 2.1 Definição

Bag of Words representa cada documento pela contagem de cada palavra do vocabulário.

```
Vocabulário: [bom, chegou, produto, horrível, excelente]

"produto excelente, chegou rápido"  →  [0, 1, 1, 0, 1]
"produto horrível, não chegou"      →  [0, 1, 1, 1, 0]
```

## 2.2 Construindo o vocabulário com FreqDist

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.probability import FreqDist

stemmer = RSLPStemmer()
stop_pt = stopwords.words('portuguese')
stop_pt.remove('não')

reviews = [
    "O produto chegou rápido e funciona perfeitamente. Adorei!",
    "Excelente qualidade. Vale cada centavo gasto no produto.",
    "Produto muito bom, bonito e resistente. Recomendo bastante.",
    "Péssimo produto. Parou de funcionar em dois dias. Lixo!",
    "Horrível. Chegou quebrado e o atendimento foi um desastre.",
    "Não funciona como anunciado. Qualidade ruim e muito frágil.",
]

# Pré-processar todas as reviews
corpus_limpo = []
for review in reviews:
    tok = word_tokenize(review.lower(), language='portuguese')
    tok_limpos = []
    for token in tok:
        if token.isalpha() and token not in stop_pt and len(token) > 2:
            tok_limpos.append(stemmer.stem(token))
    corpus_limpo.append(tok_limpos)

# Construir vocabulário
todos_tokens = []
for lista in corpus_limpo:
    for token in lista:
        todos_tokens.append(token)

freq_corpus = FreqDist(todos_tokens)
vocabulario = sorted(freq_corpus.keys())

print('Vocabulário:', vocabulario)
print('Total de palavras únicas:', len(vocabulario))
```

## 2.3 Transformar reviews em vetores

```python
# Criar vetor BoW de cada review usando FreqDist
matriz_bow = []

for lista_tokens in corpus_limpo:
    freq  = FreqDist(lista_tokens)
    vetor = []
    for palavra in vocabulario:
        vetor.append(freq[palavra])
    matriz_bow.append(vetor)

# Exibir a Matriz Documento-Termo
print('Matriz Documento-Termo (Bag of Words):')
print()

cabecalho = f'{"":>8}'
for palavra in vocabulario:
    cabecalho = cabecalho + f'{palavra:>12}'
print(cabecalho)
print('-' * (8 + 12 * len(vocabulario)))

rotulos = ['pos','pos','pos','neg','neg','neg']
for i in range(len(matriz_bow)):
    emoji = '✅' if rotulos[i] == 'pos' else '❌'
    linha = f'{emoji} Rev{i+1}'
    for valor in matriz_bow[i]:
        linha = linha + f'{valor:>12}'
    print(linha)
```

## 2.4 Esparsidade

```python
total = 0
zeros = 0

for linha in matriz_bow:
    for valor in linha:
        total = total + 1
        if valor == 0:
            zeros = zeros + 1

print(f'Total de células: {total}')
print(f'Células zero:     {zeros}')
print(f'Esparsidade:      {zeros * 100 // total}%')
```

Em corpora reais com 100.000 documentos, a esparsidade costuma passar de **99%**.

## 2.5 Limitações do BoW

| Limitação | Descrição |
|-----------|-----------|
| **Sem ordem** | `'o gato comeu o rato'` = `'o rato comeu o gato'` |
| **Sem semântica** | `'carro'` e `'automóvel'` são dimensões independentes |
| **Palavras comuns dominam** | `'produto'` aparece em tudo — alto peso, pouca informação |
| **Tamanhos diferentes** | Doc com 1.000 palavras tem contagens maiores que doc com 100 |

---

# PARTE 3 — TF-IDF

## 3.1 O problema do BoW com palavras comuns

No BoW, `'produto'` aparece em todas as reviews — mas não ajuda a
distinguir positivas de negativas. O TF-IDF penaliza automaticamente
essas palavras comuns.

## 3.2 A fórmula

```
TF(t, d)    = contagem(t, d) / total de tokens em d
IDF(t, D)   = log( N / df(t) )
TF-IDF(t,d) = TF(t, d) × IDF(t, D)

Onde:
  N     = número total de documentos
  df(t) = número de documentos que contêm o termo t
```

### Interpretação

| Situação | TF-IDF |
|----------|--------|
| Freq. alta neste doc + rara no corpus | **Alto** ✅ — muito relevante |
| Freq. alta neste doc + comum no corpus | **Baixo** ❌ — pouco discriminativo |
| Aparece em todos os docs | **Zero** — IDF = log(N/N) = 0 |

## 3.3 Exemplo numérico

Corpus de 6 reviews. Palavra `'produt'` (radical de produto):

```
df('produt') = 6  (aparece em todos os 6 documentos)
IDF('produt') = log(6 / (6+1)) = log(0,857) = -0,154   ← próximo de zero!

Palavra 'quebr' (radical de quebrado):
df('quebr') = 1  (aparece só na Review 5)
IDF('quebr') = log(6 / (1+1)) = log(3) = 1,099   ← valor alto!
```

## 3.4 Calculando TF-IDF com NLTK

```python
import math

# Calcular IDF para cada palavra do vocabulário
N   = len(corpus_limpo)
idf = {}

for palavra in vocabulario:
    df = 0
    for lista in corpus_limpo:
        if palavra in lista:
            df = df + 1
    idf[palavra] = math.log(N / (df + 1))

# Calcular TF-IDF de cada review
print('=== TF-IDF por review ===')
print()

for i in range(len(corpus_limpo)):
    tokens_i = corpus_limpo[i]
    freq_i   = FreqDist(tokens_i)
    total_i  = len(tokens_i)

    tfidf_i = {}
    for palavra in tokens_i:
        tf            = freq_i[palavra] / total_i
        tfidf_i[palavra] = tf * idf[palavra]

    top = sorted(tfidf_i, key=lambda p: tfidf_i[p], reverse=True)

    emoji = '✅' if rotulos[i] == 'pos' else '❌'
    print(f'{emoji} Review {i+1}: palavras mais relevantes (TF-IDF):')
    for palavra in top[:3]:
        print(f'   {palavra:<18} {tfidf_i[palavra]:.4f}')
    print()
```

## 3.5 BoW vs. TF-IDF para a palavra 'produto'

```python
alvo = 'produt'   # radical de 'produto'

print(f'=== "{alvo}" — BoW vs. TF-IDF ===')
print(f'  {"":>8} {"BoW":>12} {"TF-IDF":>12}')
print('  ' + '-' * 34)

for i in range(len(corpus_limpo)):
    tokens_i = corpus_limpo[i]
    freq_i   = FreqDist(tokens_i)
    total_i  = len(tokens_i)

    bow_val = freq_i[alvo]

    if total_i > 0 and alvo in tokens_i:
        tf       = freq_i[alvo] / total_i
        tfidf_v  = tf * idf.get(alvo, 0)
    else:
        tfidf_v = 0.0

    emoji = '✅' if rotulos[i] == 'pos' else '❌'
    print(f'  {emoji} Rev{i+1:<5} {bow_val:>12} {tfidf_v:>12.4f}')

print()
print(f'IDF de "{alvo}": {idf.get(alvo, 0):.4f}')
print('→ Palavra em quase todos os docs = IDF baixo = peso baixo')
```

---

# PARTE 4 — Content-Word Filtering com spaCy

## 4.1 Filtrar por classe gramatical antes de vetorizar

O spaCy permite usar apenas as classes que carregam mais significado:

```python
import spacy

nlp = spacy.load('pt_core_news_sm')

classes_conteudo = ['NOUN', 'VERB', 'ADJ', 'ADV']

corpus_spacy = []
for review in reviews:
    doc   = nlp(review.lower())
    lemas = []
    for token in doc:
        if token.pos_ in classes_conteudo and token.is_alpha and len(token.text) > 2:
            lemas.append(token.lemma_)
    corpus_spacy.append(lemas)

print('Reviews pré-processadas com spaCy (lemas de conteúdo):')
for i in range(len(corpus_spacy)):
    emoji = '✅' if rotulos[i] == 'pos' else '❌'
    print(f'  {emoji} {corpus_spacy[i]}')
```

## 4.2 TF-IDF com lemas do spaCy

```python
# Construir vocabulário spaCy
todos_lemas = []
for lista in corpus_spacy:
    for lema in lista:
        todos_lemas.append(lema)

freq_spacy_corpus = FreqDist(todos_lemas)
vocab_spacy       = sorted(freq_spacy_corpus.keys())

# Calcular IDF
idf_spacy = {}
for lema in vocab_spacy:
    df = 0
    for lista in corpus_spacy:
        if lema in lista:
            df = df + 1
    idf_spacy[lema] = math.log(len(corpus_spacy) / (df + 1))

# TF-IDF com lemas
print('=== TF-IDF com lemas spaCy ===')
print()

for i in range(len(corpus_spacy)):
    lemas_i  = corpus_spacy[i]
    freq_i   = FreqDist(lemas_i)
    total_i  = len(lemas_i)

    tfidf_i = {}
    for lema in lemas_i:
        tf            = freq_i[lema] / total_i if total_i > 0 else 0
        tfidf_i[lema] = tf * idf_spacy[lema]

    top = sorted(tfidf_i, key=lambda p: tfidf_i[p], reverse=True)

    emoji = '✅' if rotulos[i] == 'pos' else '❌'
    print(f'{emoji} Review {i+1}: {[p for p in top[:3]]}')
```

---

# PARTE 5 — Comparação das Abordagens

## 5.1 Resumo comparativo

```python
print('=== Comparação final ===')
print()

for i in range(len(reviews)):
    emoji = '✅' if rotulos[i] == 'pos' else '❌'
    print(f'{emoji} Review {i+1}: "{reviews[i][:45]}..."')

    # NLTK + BoW
    tokens_nltk = corpus_limpo[i]
    print(f'   NLTK/BoW:   {tokens_nltk}')

    # TF-IDF top
    freq_i  = FreqDist(tokens_nltk)
    total_i = len(tokens_nltk)
    tfidf_i = {}
    for p in tokens_nltk:
        tfidf_i[p] = (freq_i[p] / total_i) * idf.get(p, 0)
    top_tfidf = sorted(tfidf_i, key=lambda p: tfidf_i[p], reverse=True)[:3]
    print(f'   TF-IDF top: {top_tfidf}')

    # spaCy lemas
    print(f'   spaCy/POS:  {corpus_spacy[i]}')
    print()
```

---

# Síntese da Aula 04

## Conceitos consolidados

| Conceito | Definição resumida |
|----------|--------------------|
| **Vetorização** | Transformar texto em lista de números para algoritmos de ML |
| **Vocabulário** | Conjunto de todas as palavras únicas do corpus |
| **Bag of Words** | Vetor com contagem de cada palavra do vocabulário |
| **Matriz Documento-Termo** | Tabela: linhas = docs, colunas = vocabulário |
| **Esparsidade** | A maioria dos valores é zero — característica natural do BoW |
| **TF** | Frequência normalizada do termo no documento |
| **IDF** | Logaritmo da raridade do termo no corpus |
| **TF-IDF** | TF × IDF — penaliza palavras comuns, valoriza específicas |
| **IDF = 0** | Palavra em todos os docs — sem valor discriminativo |
| **FreqDist** | Método NLTK para contar frequência de tokens |
| **Content-word filtering** | Usar só NOUN, VERB, ADJ, ADV via `token.pos_` |

## Fórmulas essenciais

```
TF(t, d)    = contagem(t, d) / total de tokens em d
IDF(t, D)   = log( N / (df(t) + 1) )   ← +1 evita divisão por zero
TF-IDF(t,d) = TF(t, d) × IDF(t, D)
```

## Métodos usados nesta aula

| Método | Biblioteca | O que faz |
|--------|-----------|-----------|
| `word_tokenize()` | NLTK | Tokeniza texto em português |
| `stopwords.words('portuguese')` | NLTK | Lista de stopwords |
| `RSLPStemmer().stem()` | NLTK | Stemming para português |
| `FreqDist(tokens)` | NLTK | Conta frequência de cada token |
| `freq[palavra]` | NLTK | Retorna 0 se palavra não existe |
| `nlp(texto)` | spaCy | Pipeline completo |
| `token.pos_` | spaCy | Classe gramatical |
| `token.lemma_` | spaCy | Lema da palavra |
| `math.log()` | Python | Logaritmo natural |

## Conexão com as próximas aulas

- **Semana 05** — N-gramas: capturar sequências (`'não funciona'` como par)
- **Semana 06** — Word Embeddings: representação densa com semântica (Word2Vec, GloVe)

---

## Referências Bibliográficas

| Código | Referência |
|--------|-----------|
| **B1** | CASELI, H. M.; NUNES, M. G. V. (org.). *Processamento de linguagem natural*. São Carlos: BPLN, 2023. **Cap. 6, pp. 149–180.** |
| **B2** | FACELI, K. et al. *Inteligência Artificial: uma abordagem de aprendizado de máquina*. Rio de Janeiro: LTC, 2025. **Cap. 5 e 8.** |
| **C3** | SICSÚ, A. L.; SAMARTINI, A.; BARTH, N. L. *Técnicas de Machine Learning*. São Paulo: Editora Blucher, 2023. **Cap. 4, pp. 67–88.** |

---

*IBM3130 · PLN · Aula 04 · 2º Semestre de 2026*
