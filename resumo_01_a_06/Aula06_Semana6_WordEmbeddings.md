# IBM3130 — Processamento de Linguagem Natural
## Aula 06 · Semana 6 · Word Embeddings — Word2Vec e GloVe

---

| | |
|---|---|
| **Disciplina** | IBM3130 — Processamento de Linguagem Natural |
| **Aula** | 06 de 14 · Semana 6 · 2º Semestre de 2026 |
| **Data** | 03/09/2026 (quinta-feira) |
| **Módulo** | Módulo 2 — Representações de Texto |
| **Pré-requisito** | Aulas 01–05 — PLN, Pré-processamento, POS Tagging, BoW, TF-IDF, N-gramas |
| **Referências** | B1 Cap. 8 (pp. 215–248) · B3 Cap. 2 · C3 Cap. 5 |

---

## Contextualização

Nas últimas aulas aprendemos a representar textos como vetores numéricos.
O Bag of Words conta palavras. O TF-IDF pondera pela raridade.
Os N-gramas capturam sequências. Mas todas essas representações têm
uma limitação fundamental em comum:

> **Elas não entendem o significado das palavras.**

Para o BoW, `'carro'` e `'automóvel'` são dois tokens completamente
diferentes — como se não tivessem nenhuma relação. O vetor de `'rei'`
não tem nenhuma proximidade com o vetor de `'rainha'`.
E `'bom'` não está perto de `'excelente'` — são apenas colunas diferentes
na mesma matriz esparsa.

Os **Word Embeddings** resolvem exatamente esse problema.
Eles representam cada palavra como um **vetor denso** em um espaço de
alta dimensão, onde palavras com **significados parecidos ficam próximas**.

Esse é um dos avanços mais importantes da história do PLN —
e é a base conceitual dos modelos modernos como o BERT e o GPT.

### O que você vai aprender nesta aula

| Conceito | O que é |
|----------|---------|
| **Representação densa** | Vetor com poucos números, todos significativos |
| **Representação esparsa** | Vetor com muitos zeros — como o BoW |
| **Word Embedding** | Vetor denso que representa o significado de uma palavra |
| **Word2Vec** | Algoritmo que aprende embeddings a partir do contexto |
| **CBOW** | Arquitetura que prevê uma palavra pelo contexto ao redor |
| **Skip-gram** | Arquitetura que prevê o contexto a partir de uma palavra |
| **GloVe** | Embeddings baseados em co-ocorrências globais do corpus |
| **Similaridade de cosseno** | Medida de proximidade entre vetores |
| **Analogias vetoriais** | Operações aritméticas com significado semântico |
| **PCA** | Técnica para visualizar vetores de alta dimensão em 2D |

---

# PARTE 1 — O Problema das Representações Esparsas

## 1.1 Revisando o Bag of Words

No BoW, cada palavra do vocabulário é uma dimensão do vetor.
Para um vocabulário de 10.000 palavras, cada texto vira um vetor
com 10.000 posições — onde a maioria é zero.

```
Vocabulário: [bom, carro, excelente, gato, ótimo, péssimo, ...]
                                            ↑ 10.000 colunas

"o carro é bom"   →  [1, 1, 0, 0, 0, 0, ...]
"o carro é ótimo" →  [0, 1, 0, 0, 1, 0, ...]
```

Esses dois vetores têm pouquíssima sobreposição — mas as frases têm
o mesmo sentido. O BoW não consegue enxergar isso.

## 1.2 Os três problemas que os embeddings resolvem

### Problema 1 — Sem semântica

No BoW, `'bom'` e `'excelente'` são dimensões completamente independentes.
Um modelo treinado com exemplos contendo `'bom'` não aprende nada
sobre `'excelente'` — mesmo que as duas palavras tenham o mesmo significado.

```
BoW:
  'bom'       →  [0, 0, 0, 1, 0, 0, 0, ...]   ← dimensão 3
  'excelente' →  [0, 0, 0, 0, 0, 1, 0, ...]   ← dimensão 5
  Distância entre elas: MÁXIMA (não têm nada em comum no vetor)

Embedding:
  'bom'       →  [0.82, -0.11, 0.45, ...]
  'excelente' →  [0.79, -0.08, 0.51, ...]
  Distância entre elas: MÍNIMA (vetores quase idênticos)
```

### Problema 2 — Maldição da dimensionalidade

Com 100.000 palavras no vocabulário, cada vetor BoW tem 100.000 dimensões.
Em espaços de alta dimensão, todos os pontos ficam igualmente distantes —
a noção de "proximidade" perde o sentido.

Com embeddings, usamos tipicamente 100 a 300 dimensões — muito mais compacto
e matematicamente tratável.

### Problema 3 — Esparsidade

99% das entradas do vetor BoW são zero. Isso desperdiça memória,
torna o treinamento lento e dificulta o aprendizado.

Embeddings são **densos** — todos os valores são números reais significativos.

---

## 1.3 A ideia central dos embeddings

A hipótese que fundamenta os word embeddings é conhecida como
**hipótese distribucional**, formulada pelo linguista John Rupert Firth em 1957:

> *"Você conhece uma palavra pelo contexto em que ela aparece."*

Em outras palavras: palavras que aparecem nos mesmos contextos tendem
a ter significados parecidos.

`'cachorro'` e `'gato'` aparecem em contextos similares:
- "meu ___ latiu / miou"
- "o ___ dorme no sofá"
- "dei comida para o ___"

Portanto, `'cachorro'` e `'gato'` devem ter vetores próximos.

Essa hipótese é o que o Word2Vec operacionaliza de forma computacional.

---

# PARTE 2 — Word2Vec

## 2.1 O que é Word2Vec?

Word2Vec é um conjunto de técnicas desenvolvido por Tomas Mikolov e equipe
no Google em 2013. Ele usa uma rede neural rasa (uma única camada oculta)
para aprender representações densas de palavras a partir de grandes corpora.

O resultado são vetores numéricos onde:
- Palavras semelhantes ficam próximas no espaço vetorial
- Relações semânticas se traduzem em direções geométricas

Word2Vec tem duas arquiteturas principais: **CBOW** e **Skip-gram**.

---

## 2.2 Arquitetura CBOW — Continuous Bag of Words

**Ideia:** dadas as palavras ao redor, prever a palavra do meio.

```
Janela de contexto = 2 palavras de cada lado

Frase: "o modelo neural aprende com os dados"

Entrada (contexto):  ["o", "modelo", "aprende", "com"]
Saída (alvo):        "neural"
```

O modelo aprende os pesos da rede para fazer essa previsão.
Os pesos da camada oculta são os **embeddings**.

### Visualizando o CBOW

```
[o]       ─┐
[modelo]  ─┤→  [camada oculta 300 dims]  →  P("neural") = 0.72
[aprende] ─┤                                P("rápido") = 0.05
[com]     ─┘                                P("lento")  = 0.01
                                            ...
```

### Características do CBOW

- Mais rápido para treinar
- Funciona melhor com corpus **grande**
- Suaviza sobre o contexto — ignora detalhes de palavras raras

---

## 2.3 Arquitetura Skip-gram

**Ideia:** dada uma palavra, prever as palavras ao redor.

```
Frase: "o modelo neural aprende com os dados"

Entrada (palavra central): "neural"
Saída (contexto):          ["o", "modelo", "aprende", "com"]
```

É o processo inverso do CBOW.

### Visualizando o Skip-gram

```
                      →  P("o")      = 0.15
                      →  P("modelo") = 0.68
["neural"]  →  [camada oculta]
                      →  P("aprende") = 0.71
                      →  P("com")     = 0.12
```

### Características do Skip-gram

- Mais lento para treinar
- Funciona melhor com corpus **pequeno**
- Captura melhor palavras raras e infrequentes
- **Preferido na maioria dos casos práticos**

---

## 2.4 CBOW vs. Skip-gram — resumo

| Aspecto | CBOW | Skip-gram |
|---------|------|-----------|
| Tarefa | Contexto → palavra central | Palavra central → contexto |
| Velocidade de treino | Mais rápido | Mais lento |
| Corpus ideal | Grande | Pequeno ou médio |
| Palavras raras | Tratamento inferior | Tratamento superior |
| Uso geral | Menos comum | **Mais popular** |

---

## 2.5 O que o Word2Vec aprende?

O Word2Vec aprende que palavras usadas em contextos similares
devem ter vetores similares. Vejamos alguns exemplos do que emerge
espontaneamente desse aprendizado:

### Similaridade semântica

```
'rei'     →  [0.82, 0.31, -0.14, 0.56, ...]
'rainha'  →  [0.79, 0.28, -0.11, 0.54, ...]   ← muito próximos!

'cachorro' →  [0.43, -0.22, 0.67, 0.11, ...]
'gato'     →  [0.41, -0.19, 0.65, 0.13, ...]  ← muito próximos!

'cachorro' vs 'computador' →  completamente distantes
```

### Analogias vetoriais

O resultado mais famoso do Word2Vec: relações semânticas se traduzem
em **operações aritméticas** com os vetores.

```
rei - homem + mulher ≈ rainha
```

Isso significa: "a relação entre rei e homem é a mesma que entre rainha e mulher."

```
Paris - França + Itália ≈ Roma
Brasil - português + espanhol ≈ Argentina
médico - homem + mulher ≈ médica
```

Essas analogias emergem **automaticamente** do treinamento — ninguém
programou essas relações explicitamente. O modelo as aprendeu a partir
dos padrões de co-ocorrência no texto.

---

## 2.6 Treinando Word2Vec com Gensim

O Gensim é a biblioteca mais usada para Word2Vec em Python.

### Instalação e treino básico

```python
# Instalar no Colab
!pip install gensim -q

from gensim.models import Word2Vec

# Corpus simples — lista de listas de tokens
corpus = [
    ["o", "gato", "dorme", "no", "sofá"],
    ["o", "cachorro", "corre", "no", "parque"],
    ["o", "gato", "come", "o", "rato"],
    ["o", "cachorro", "late", "muito"],
    ["o", "gato", "e", "o", "cachorro", "brincam"],
    ["o", "rato", "corre", "do", "gato"],
    ["o", "cachorro", "dorme", "no", "chão"],
    ["o", "gato", "late", "igual", "ao", "cachorro"],
]

# Treinar o modelo Word2Vec
modelo = Word2Vec(
    sentences = corpus,    # corpus — lista de listas de tokens
    vector_size = 10,      # dimensões do embedding (baixo para exemplo didático)
    window = 2,            # janela de contexto — palavras de cada lado
    min_count = 1,         # mínimo de ocorrências para incluir a palavra
    sg = 1,                # 1 = Skip-gram, 0 = CBOW
    epochs = 100,          # número de passagens pelo corpus
    seed = 42              # para reprodutibilidade
)

print("Modelo treinado!")
print("Vocabulário:", list(modelo.wv.key_to_index.keys()))
print("Dimensões do embedding:", modelo.wv.vector_size)
```

### Acessando os vetores

```python
# Ver o vetor de uma palavra
vetor_gato = modelo.wv["gato"]
print("Vetor de 'gato':")
print(vetor_gato)
print("Formato:", vetor_gato.shape)

# Ver o vetor de outra palavra
vetor_cachorro = modelo.wv["cachorro"]
print("\nVetor de 'cachorro':")
print(vetor_cachorro)
```

### Palavras mais similares

```python
# Encontrar palavras mais parecidas com 'gato'
similares_gato = modelo.wv.most_similar("gato", topn=3)

print("Palavras mais similares a 'gato':")
for palavra, similaridade in similares_gato:
    print(f"  {palavra:<15} similaridade: {similaridade:.4f}")

# Fazer o mesmo para 'cachorro'
similares_cachorro = modelo.wv.most_similar("cachorro", topn=3)

print("\nPalavras mais similares a 'cachorro':")
for palavra, similaridade in similares_cachorro:
    print(f"  {palavra:<15} similaridade: {similaridade:.4f}")
```

---

# PARTE 3 — Similaridade de Cosseno

## 3.1 Como medir a proximidade entre vetores?

Dado que cada palavra é um vetor, precisamos de uma forma de medir
**o quão próximos** dois vetores estão. A medida mais usada em PLN é a
**similaridade de cosseno**.

## 3.2 A ideia geométrica

A similaridade de cosseno não mede a **distância** entre dois pontos,
mas o **ângulo** entre os dois vetores:

```
Vetores paralelos (ângulo = 0°)    →  cosseno = 1,0  → máxima similaridade
Vetores perpendiculares (ângulo = 90°) →  cosseno = 0,0  → sem relação
Vetores opostos (ângulo = 180°)    →  cosseno = -1,0 → significados opostos
```

Isso é importante porque palavras com contextos similares tendem a
apontar para a **mesma direção** no espaço vetorial — mesmo que
tenham magnitudes diferentes.

## 3.3 A fórmula

```
                A · B
cos(A, B)  =  ─────────
              |A| × |B|

Onde:
  A · B  = produto escalar (soma dos produtos elemento a elemento)
  |A|    = norma do vetor A (raiz quadrada da soma dos quadrados)
  |B|    = norma do vetor B
```

## 3.4 Exemplo numérico completo

```python
import math

# Vetores simplificados de duas palavras (apenas 3 dimensões para facilitar)
vetor_gato     = [0.8, 0.3, 0.5]
vetor_cachorro = [0.7, 0.4, 0.6]
vetor_carro    = [0.1, 0.9, 0.2]

def similaridade_cosseno(v1, v2):

    # Passo 1: produto escalar (A · B)
    produto = 0
    for i in range(len(v1)):
        produto = produto + v1[i] * v2[i]

    # Passo 2: norma do vetor v1 (|A|)
    soma_quadrados_v1 = 0
    for x in v1:
        soma_quadrados_v1 = soma_quadrados_v1 + x * x
    norma_v1 = math.sqrt(soma_quadrados_v1)

    # Passo 3: norma do vetor v2 (|B|)
    soma_quadrados_v2 = 0
    for x in v2:
        soma_quadrados_v2 = soma_quadrados_v2 + x * x
    norma_v2 = math.sqrt(soma_quadrados_v2)

    # Passo 4: dividir
    return produto / (norma_v1 * norma_v2)

# Calcular similaridades
sim_gato_cachorro = similaridade_cosseno(vetor_gato, vetor_cachorro)
sim_gato_carro    = similaridade_cosseno(vetor_gato, vetor_carro)

print("=== Similaridade de Cosseno ===")
print()
print(f"  'gato' vs 'cachorro' = {sim_gato_cachorro:.4f}")
print(f"  'gato' vs 'carro'    = {sim_gato_carro:.4f}")
print()

if sim_gato_cachorro > sim_gato_carro:
    print("  → 'cachorro' é mais similar a 'gato' do que 'carro'")
    print("    (faz sentido — ambos são animais de estimação!)")
```

## 3.5 Similaridade com Gensim

O Gensim calcula a similaridade de cosseno automaticamente:

```python
# Similaridade entre duas palavras usando Gensim
sim = modelo.wv.similarity("gato", "cachorro")
print(f"Similaridade('gato', 'cachorro') = {sim:.4f}")

sim2 = modelo.wv.similarity("gato", "rato")
print(f"Similaridade('gato', 'rato')     = {sim2:.4f}")

sim3 = modelo.wv.similarity("gato", "sofá")
print(f"Similaridade('gato', 'sofá')     = {sim3:.4f}")

# Qual par é mais similar?
pares = [
    ("gato", "cachorro"),
    ("gato", "rato"),
    ("gato", "sofá"),
    ("cachorro", "rato"),
]

print("\nRanking de similaridade:")
resultados = []
for p1, p2 in pares:
    sim = modelo.wv.similarity(p1, p2)
    resultados.append((p1, p2, sim))

resultados.sort(key=lambda x: x[2], reverse=True)

posicao = 1
for p1, p2, sim in resultados:
    print(f"  {posicao}. '{p1}' vs '{p2}': {sim:.4f}")
    posicao = posicao + 1
```

---

# PARTE 4 — Analogias Vetoriais

## 4.1 O que são analogias vetoriais?

Uma das descobertas mais surpreendentes do Word2Vec foi que
**relações semânticas se traduzem em direções geométricas** consistentes.

Se o modelo aprendeu que `'rei'` é para `'homem'` assim como `'rainha'`
é para `'mulher'`, então geometricamente:

```
vetor("rei") - vetor("homem") + vetor("mulher") ≈ vetor("rainha")
```

Isso funciona porque o modelo aprendeu, implicitamente, que existe uma
direção no espaço vetorial que representa o conceito de "gênero feminino vs. masculino".

## 4.2 Como funciona na prática

```python
# Usando Gensim — positive = palavras a somar, negative = palavras a subtrair
resultado = modelo.wv.most_similar(
    positive=["rainha", "homem"],
    negative=["rei"],
    topn=3
)

# Equivale a: rainha - rei + homem ≈ ?
print("rainha - rei + homem ≈ ?")
for palavra, similaridade in resultado:
    print(f"  {palavra}  ({similaridade:.4f})")
```

## 4.3 Exemplos famosos de analogias

Esses exemplos foram demonstrados no corpus Google News (100 bilhões de palavras):

```
# Relações de gênero
rei   - homem  + mulher  ≈ rainha
ator  - homem  + mulher  ≈ atriz

# Capitais de países
Paris - França + Itália  ≈ Roma
Paris - França + Brasil  ≈ Brasília

# Formas verbais
correr - correndo + andando ≈ andar

# Comparativos
bom - melhor + pior ≈ ruim
```

## 4.4 Por que isso acontece?

Imagine o espaço vetorial como um mapa. Cada palavra é um ponto nesse mapa.

Se você "caminhar" do ponto `'homem'` ao ponto `'mulher'`, percorrerá
uma distância em uma direção específica. Essa mesma distância e direção,
aplicada a partir do ponto `'rei'`, chega perto do ponto `'rainha'`.

```
          mulher ──────────────→ rainha
            ↑                      ↑
    (mesma  │                      │ (mesma
   direção) │                      │ distância)
            │                      │
          homem  ──────────────→  rei
```

---

# PARTE 5 — Embeddings Pré-treinados

## 5.1 O problema do corpus pequeno

O Word2Vec precisa de muito texto para aprender boas representações.
Com corpus pequeno, os vetores ficam ruins — o modelo não vê contextos suficientes.

A solução: usar embeddings **pré-treinados** em corpus massivos.

## 5.2 Embeddings disponíveis publicamente

| Modelo | Corpus de treino | Dimensões | Tamanho |
|--------|-----------------|-----------|---------|
| Word2Vec (Google News) | 100 bilhões de palavras (inglês) | 300 | 1,5 GB |
| GloVe (Wikipedia + Gigaword) | 6 bilhões de tokens (inglês) | 50–300 | 822 MB |
| fastText (Wikipedia) | Múltiplos idiomas | 300 | ~6 GB |
| NILC/USP (português) | Corpus brasileiro | 50–600 | Variado |

## 5.3 Usando embeddings pré-treinados com Gensim

```python
import gensim.downloader as api

# Baixar embeddings pré-treinados (pode demorar — arquivo grande)
# 'glove-wiki-gigaword-50' é o menor disponível: 50 dimensões
modelo_pt = api.load("glove-wiki-gigaword-50")

print("Modelo carregado!")
print("Vocabulário:", len(modelo_pt.key_to_index), "palavras")
print("Dimensões:", modelo_pt.vector_size)

# Testar similaridade
print("\nSimilaridade 'king' vs 'queen':", modelo_pt.similarity("king", "queen"))
print("Similaridade 'cat' vs 'dog':   ", modelo_pt.similarity("cat",  "dog"))
print("Similaridade 'cat' vs 'car':   ", modelo_pt.similarity("cat",  "car"))

# Analogia: king - man + woman = ?
resultado = modelo_pt.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=3
)
print("\nking - man + woman ≈")
for palavra, sim in resultado:
    print(f"  {palavra} ({sim:.4f})")
```

## 5.4 Embeddings para português — NILC/USP

Para projetos em português, o NILC (USP) disponibiliza embeddings
treinados em corpus brasileiro:

```python
# Baixar embeddings do NILC — exemplo com skip-gram 50 dimensões
# Link: http://nilc.icmc.usp.br/embeddings

# Após baixar o arquivo .txt:
from gensim.models import KeyedVectors

modelo_pt = KeyedVectors.load_word2vec_format(
    'skip_s50.txt',    # arquivo baixado do NILC
    binary=False
)

print("Palavras similares a 'inteligência' em português:")
for palavra, sim in modelo_pt.most_similar("inteligência", topn=5):
    print(f"  {palavra:<20} {sim:.4f}")
```

---

# PARTE 6 — GloVe

## 6.1 O que é GloVe?

**GloVe** (Global Vectors for Word Representation) foi desenvolvido por
Jeffrey Pennington, Richard Socher e Christopher Manning em Stanford em 2014.

Enquanto o Word2Vec aprende embeddings a partir de **janelas de contexto locais**
(palavras vizinhas), o GloVe usa **estatísticas globais de co-ocorrência** —
conta quantas vezes cada par de palavras aparece juntas em todo o corpus.

## 6.2 A intuição do GloVe

O GloVe parte de uma pergunta: o que a proporção entre as probabilidades
de co-ocorrência de palavras nos diz sobre seus significados?

```
Considere as palavras: "gelo", "água", "sólido", "gás"

P("sólido" | "gelo")    é ALTA   — gelo é sólido
P("sólido" | "água")    é BAIXA  — água não é sólida (em geral)

P("gás"    | "gelo")    é BAIXA
P("gás"    | "água")    é ALTA   — água pode virar gás (vapor)

A razão:
P("sólido"|"gelo") / P("sólido"|"água")  →  valor ALTO
P("gás"   |"gelo") / P("gás"   |"água")  →  valor BAIXO
```

Essas **proporções** capturam relações semânticas de forma mais robusta
do que probabilidades individuais.

## 6.3 GloVe vs. Word2Vec

| Aspecto | Word2Vec | GloVe |
|---------|----------|-------|
| Base | Janelas de contexto locais | Co-ocorrências globais |
| Treinamento | Gradiente estocástico | Mínimos quadrados ponderados |
| Corpus necessário | Passa várias vezes | Uma vez (matriz de co-ocorrência) |
| Analogias | Muito bom | Muito bom |
| Palavras raras | Skip-gram é superior | Inferior |
| Velocidade | Mais lento | Mais rápido |
| Uso atual | Muito popular | Muito popular |

Na prática, os dois produzem resultados muito similares.
A escolha geralmente depende dos recursos computacionais disponíveis.

## 6.4 Usando GloVe com Gensim

```python
# O Gensim pode carregar embeddings no formato GloVe
# Após baixar o arquivo glove.6B.50d.txt de nlp.stanford.edu/projects/glove/

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Converter formato GloVe para formato Word2Vec
glove2word2vec("glove.6B.50d.txt", "glove_w2v.txt")

# Carregar
glove = KeyedVectors.load_word2vec_format("glove_w2v.txt", binary=False)

# Usar normalmente
print(glove.most_similar("computer", topn=5))
print(glove.similarity("good", "excellent"))
```

---

# PARTE 7 — Visualizando Embeddings com PCA

## 7.1 O problema de visualizar alta dimensão

Um embedding típico tem 100 a 300 dimensões.
Não é possível visualizar um espaço de 300 dimensões diretamente.

A solução: usar **PCA** (Principal Component Analysis) para comprimir
os vetores para apenas 2 dimensões, preservando ao máximo as distâncias originais.

## 7.2 O que é PCA?

PCA encontra as duas direções no espaço de alta dimensão que mais
"explicam" a variação dos dados — e projeta todos os pontos nessas direções.

```
300 dimensões  →  PCA  →  2 dimensões  →  gráfico de dispersão
```

É como fotografar um objeto 3D: você perde uma dimensão, mas consegue
ver a estrutura geral.

## 7.3 Visualizando Word2Vec com PCA

```python
import math
from gensim.models import Word2Vec

# Corpus sobre tecnologia e animais — para ver grupos distintos
corpus = [
    ["inteligência", "artificial", "aprende", "dados"],
    ["rede", "neural", "processa", "texto"],
    ["algoritmo", "machine", "learning", "treina"],
    ["modelo", "linguagem", "texto", "palavras"],
    ["computador", "processa", "dados", "rápido"],
    ["gato", "dorme", "sofá", "ronrona"],
    ["cachorro", "late", "corre", "brinca"],
    ["pássaro", "voa", "canta", "ninho"],
    ["peixe", "nada", "água", "aquário"],
    ["coelho", "pula", "come", "cenoura"],
]

# Treinar Word2Vec
modelo = Word2Vec(
    sentences    = corpus,
    vector_size  = 20,
    window       = 2,
    min_count    = 1,
    sg           = 1,
    epochs       = 200,
    seed         = 42
)

# Palavras que vamos visualizar
palavras_tecnologia = ["inteligência", "rede", "algoritmo", "modelo", "computador"]
palavras_animais    = ["gato", "cachorro", "pássaro", "peixe", "coelho"]
palavras_todas      = palavras_tecnologia + palavras_animais

# Extrair vetores
vetores = []
for palavra in palavras_todas:
    if palavra in modelo.wv:
        vetores.append(modelo.wv[palavra].tolist())

print(f"Total de vetores: {len(vetores)}")
print(f"Dimensões originais: {len(vetores[0])}")
```

```python
# Redução de dimensionalidade manual com PCA simplificado
# (sem sklearn — usando apenas operações básicas)

# Passo 1: Centralizar os vetores (subtrair a média)
n_palavras = len(vetores)
n_dims     = len(vetores[0])

# Calcular a média de cada dimensão
medias = []
for d in range(n_dims):
    soma = 0
    for v in vetores:
        soma = soma + v[d]
    medias.append(soma / n_palavras)

# Subtrair a média
vetores_centralizados = []
for v in vetores:
    v_cent = []
    for d in range(n_dims):
        v_cent.append(v[d] - medias[d])
    vetores_centralizados.append(v_cent)

print("Vetores centralizados (primeiros 3 valores do vetor 0):")
print(vetores_centralizados[0][:3])
```

```python
import matplotlib.pyplot as plt

# Usando sklearn apenas para o PCA (não há forma simples sem ele)
from sklearn.decomposition import PCA

import numpy as np

# Converter para array numpy
X = np.array(vetores)

# Aplicar PCA — reduzir para 2 dimensões
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plotar
fig, ax = plt.subplots(figsize=(10, 7))

# Plotar tecnologia em azul
for i in range(len(palavras_tecnologia)):
    ax.scatter(X_2d[i, 0], X_2d[i, 1], color='#2E75B6', s=100)
    ax.annotate(palavras_todas[i],
                xy=(X_2d[i, 0], X_2d[i, 1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=11, color='#2E75B6')

# Plotar animais em verde
for i in range(len(palavras_tecnologia), len(palavras_todas)):
    ax.scatter(X_2d[i, 0], X_2d[i, 1], color='#1A5E3A', s=100)
    ax.annotate(palavras_todas[i],
                xy=(X_2d[i, 0], X_2d[i, 1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=11, color='#1A5E3A')

ax.set_title('Word Embeddings visualizados em 2D (PCA)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.grid(True, alpha=0.3)

# Legenda manual
from matplotlib.patches import Patch
legenda = [
    Patch(color='#2E75B6', label='Tecnologia / IA'),
    Patch(color='#1A5E3A', label='Animais'),
]
ax.legend(handles=legenda, fontsize=11)

plt.tight_layout()
plt.show()

print("Observe como palavras do mesmo campo semântico")
print("ficam agrupadas no gráfico!")
```

## 7.4 O que esperar do gráfico

Se os embeddings foram bem aprendidos, você deve ver:
- Palavras de tecnologia (`'rede'`, `'algoritmo'`, `'modelo'`) **agrupadas**
- Palavras de animais (`'gato'`, `'cachorro'`, `'peixe'`) em **outro grupo**
- Os dois grupos **separados** no espaço 2D

Com um corpus tão pequeno (10 frases), os grupos podem não estar perfeitos —
mas com um corpus maior, a separação fica muito clara.

---

# PARTE 8 — Word Embeddings em PLN Moderno

## 8.1 Embeddings estáticos vs. contextualizados

Os embeddings que vimos até agora são **estáticos**: cada palavra tem
um único vetor, independentemente do contexto em que aparece.

```
"banco" em "fui ao banco sacar dinheiro"  →  mesmo vetor
"banco" em "sentei no banco da praça"     →  mesmo vetor  ← problema!
```

Os modelos modernos usam **embeddings contextualizados**: o vetor de uma
palavra muda dependendo da frase em que ela aparece.

| Tipo | Exemplos | Característica |
|------|---------|---------------|
| **Estático** | Word2Vec, GloVe, fastText | Um vetor por palavra |
| **Contextualizado** | BERT, GPT, RoBERTa | Vetor muda com o contexto |

## 8.2 A evolução dos embeddings

```
Representação one-hot (BoW)
    ↓  sem semântica
Word2Vec / GloVe (2013–2014)
    ↓  semântica estática
ELMo (2018)  →  embeddings com contexto (LSTMs)
    ↓  ainda sequencial
BERT (2018)  →  contexto bidirecional completo (Transformers)
    ↓  ainda maior
GPT-3, GPT-4, Claude (2020+)  →  LLMs com bilhões de parâmetros
```

## 8.3 Aplicações práticas de embeddings

```python
# Aplicação 1: busca semântica
# Encontrar documentos semanticamente similares à query

query    = "inteligência artificial aprende"
docs     = ["machine learning processa dados",
            "gato dorme no sofá",
            "rede neural treina modelo"]

# Vetorizar query e documentos com word2vec
# (média dos vetores das palavras)
def media_vetores(tokens, modelo):
    vetores_validos = []
    for token in tokens:
        if token in modelo.wv:
            vetores_validos.append(modelo.wv[token])
    if not vetores_validos:
        return None
    # Calcular média manualmente
    n    = len(vetores_validos)
    dims = len(vetores_validos[0])
    media = []
    for d in range(dims):
        soma = 0
        for v in vetores_validos:
            soma = soma + v[d]
        media.append(soma / n)
    return media

from nltk.tokenize import word_tokenize

tokens_query = word_tokenize(query.lower(), language='portuguese')
vetor_query  = media_vetores(tokens_query, modelo)

print("Busca semântica para:", query)
print()

resultados = []
for doc in docs:
    tokens_doc  = word_tokenize(doc.lower(), language='portuguese')
    vetor_doc   = media_vetores(tokens_doc, modelo)
    if vetor_doc and vetor_query:
        sim = similaridade_cosseno(vetor_query, vetor_doc)
        resultados.append((doc, sim))

resultados.sort(key=lambda x: x[1], reverse=True)

for doc, sim in resultados:
    print(f"  {sim:.4f}  '{doc}'")
```

---

# Síntese da Aula 06

## Conceitos consolidados

| Conceito | Definição resumida |
|----------|--------------------|
| **Representação densa** | Vetor com todas as posições preenchidas com valores reais |
| **Hipótese distribucional** | Palavras que aparecem em contextos similares têm significados similares |
| **Word2Vec** | Algoritmo que aprende embeddings a partir de janelas de contexto |
| **CBOW** | Prevê a palavra central dado o contexto ao redor |
| **Skip-gram** | Prevê o contexto dado a palavra central |
| **GloVe** | Embeddings baseados em co-ocorrências globais do corpus |
| **Similaridade de cosseno** | Mede o ângulo entre dois vetores — entre -1 e 1 |
| **Analogia vetorial** | `rei - homem + mulher ≈ rainha` — relações semânticas como aritmética |
| **PCA** | Reduz dimensionalidade para visualização em 2D |
| **Embedding estático** | Um único vetor por palavra, independente do contexto |
| **Embedding contextualizado** | Vetor muda conforme o contexto (BERT, GPT) |

## Fórmulas essenciais

```
# Similaridade de cosseno
             A · B
cos(A,B) = ─────────
            |A| × |B|

# Produto escalar (A · B)
A · B = A[0]*B[0] + A[1]*B[1] + ... + A[n]*B[n]

# Norma de um vetor (|A|)
|A| = sqrt(A[0]² + A[1]² + ... + A[n]²)

# Analogia vetorial
resultado = vetor(A) - vetor(B) + vetor(C)
ex:  vetor("rei") - vetor("homem") + vetor("mulher") ≈ vetor("rainha")

# Vetor médio de uma frase
vetor_frase = (v1 + v2 + ... + vn) / n
```

## Comparativo das representações

| Representação | Dimensões | Semântica | Ordem | Contexto |
|---------------|-----------|-----------|-------|---------|
| BoW | V (10k–100k) | ❌ | ❌ | ❌ |
| TF-IDF | V (10k–100k) | ❌ | ❌ | ❌ |
| N-gramas | V² (enorme) | ❌ | Parcial | ❌ |
| Word2Vec / GloVe | 50–300 | ✅ | ❌ | ❌ |
| BERT | 768 | ✅ | ✅ | ✅ |

## Conexão com as próximas aulas

- **Semana 07 — Representações conceituais:** WordNet e grafos de conhecimento —
  representação simbólica (regras e ontologias) vs. representação distribuída (embeddings)
- **Semana 10 — Redes Neurais para PLN:** RNN e LSTM que usam embeddings como entrada
- **Semana 11 — Transformers:** BERT e GPT — embeddings contextualizados com atenção global

---

## Referências Bibliográficas

| Código | Referência |
|--------|-----------|
| **B1** | CASELI, H. M.; NUNES, M. G. V. (org.). *Processamento de linguagem natural: conceitos, técnicas e aplicações em português*. São Carlos: BPLN, 2023. **Cap. 8, pp. 215–248.** |
| **B3** | SUMATHI, S.; JANANI, M. *Neural Networks for Natural Language Processing*. Hershey, PA: Engineering Science Reference, 2020. **Cap. 2, pp. 28–65.** |
| **C3** | SICSÚ, A. L.; SAMARTINI, A.; BARTH, N. L. *Técnicas de Machine Learning*. São Paulo: Editora Blucher, 2023. **Cap. 5, pp. 89–118.** |

### Recursos online gratuitos

- **Word2Vec original (Mikolov et al., 2013):**
  [arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
- **GloVe (Pennington et al., 2014):**
  [nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- **Gensim Word2Vec:**
  [radimrehurek.com/gensim/models/word2vec.html](https://radimrehurek.com/gensim/models/word2vec.html)
- **NILC — Embeddings para português:**
  [nilc.icmc.usp.br/embeddings](http://nilc.icmc.usp.br/embeddings)
- **Embedding Projector (Google):**
  [projector.tensorflow.org](https://projector.tensorflow.org) — visualização interativa de embeddings

---

*IBM3130 · PLN · Aula 06 · 2º Semestre de 2026*
