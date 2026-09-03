# IBM3130 — Processamento de Linguagem Natural
## Aula 03 · Semana 3 · Morfossintaxe e POS Tagging

---

| | |
|---|---|
| **Disciplina** | IBM3130 — Processamento de Linguagem Natural |
| **Aula** | 03 de 14 · Semana 3 · 2º Semestre de 2026 |
| **Módulo** | Módulo 1 — Fundamentos |
| **Pré-requisito** | Aulas 01–02 — PLN e Pré-processamento |
| **Ferramentas** | Google Colab · Python · NLTK · spaCy · displacy |
| **Referências** | B1 Cap. 4–5 (pp. 85–148) · C1 Cap. 3 |

---

## Contextualização

Na Aula 02 aprendemos a limpar textos: tokenizar, normalizar, remover stopwords
e aplicar stemming ou lematização. Temos agora sequências de tokens limpos.

Mas um token limpo ainda não carrega informação sobre **o que ele é**
dentro da frase. A palavra `'banco'` é um substantivo ou um verbo?
`'modelo'` é um objeto ou uma ação? `'rápido'` descreve um substantivo ou um verbo?

Responder a essas perguntas é a tarefa do **POS Tagging** — marcação morfossintática.

---

# PARTE 1 — Morfologia e Classes Gramaticais

## 1.1 Morfemas — os blocos de construção das palavras

O morfema é a menor unidade com significado em uma língua.
Identificar morfemas ajuda o computador a entender que palavras
de aparência diferente compartilham a mesma raiz.

| Palavra | Decomposição morfológica |
|---------|--------------------------|
| `'infelizmente'` | `in` (negação) + `feliz` (raiz) + `mente` (modo) |
| `'processamento'` | `process` (raiz) + `amento` (nominalizador) |
| `'pesquisadora'` | `pesquis` (raiz) + `ador` (agentivo) + `a` (feminino) |
| `'pré-processamento'` | `pré` (antes) + `process` (raiz) + `amento` (nominalizador) |

## 1.2 As dez classes gramaticais tradicionais

A gramática tradicional portuguesa distingue dez classes:

| # | Classe | Tag UD | Descrição | Exemplos |
|---|--------|--------|-----------|---------|
| 1 | **Substantivo** | NOUN / PROPN | Nomeia seres, objetos, conceitos | modelo, Google, algoritmo |
| 2 | **Artigo** | DET | Determina e acompanha substantivos | o, a, um, uma |
| 3 | **Adjetivo** | ADJ | Qualifica o substantivo | rápido, neural, excelente |
| 4 | **Numeral** | NUM | Indica quantidade ou ordem | dois, primeiro, 42 |
| 5 | **Pronome** | PRON | Substitui ou acompanha o substantivo | eu, meu, este, que |
| 6 | **Verbo** | VERB / AUX | Exprime ação, estado ou processo | correr, ser, classificou |
| 7 | **Advérbio** | ADV | Modifica verbo, adjetivo ou advérbio | muito, não, rapidamente |
| 8 | **Preposição** | ADP | Liga palavras — relações de lugar, tempo etc. | de, em, para, com |
| 9 | **Conjunção** | CCONJ / SCONJ | Liga orações ou termos | e, ou, mas, porque |
| 10 | **Interjeição** | INTJ | Expressa emoção espontânea | Ah!, Uau!, Ei! |

---

# PARTE 2 — Tagsets

## 2.1 Penn Treebank — o padrão do NLTK

O Penn Treebank (PTB) foi desenvolvido na Universidade da Pensilvânia em 1992.
Seu tagset de 48 rótulos é o padrão para o inglês e é usado pelo NLTK.

| Tag PTB | Classe | Exemplo (inglês) |
|---------|--------|-----------------|
| NN | Subs. singular | model, data |
| NNS | Subs. plural | models, datasets |
| NNP | Nome próprio | Google, Python |
| VB | Verbo base | train, run |
| VBD | Verbo passado | trained, ran |
| VBG | Gerúndio | training, running |
| VBN | Particípio passado | trained, processed |
| JJ | Adjetivo | fast, accurate |
| JJR | Adjetivo comparativo | faster, better |
| RB | Advérbio | quickly, not, very |
| IN | Preposição | in, of, for |
| CC | Conjunção coord. | and, or, but |
| DT | Determinante | the, a, an |
| PRP | Pronome pessoal | I, he, she |
| CD | Numeral | 1, 42, first |
| MD | Modal | can, will, must |

## 2.2 Universal Dependencies — o padrão do spaCy

O Universal Dependencies (UD), iniciado em 2014, propôs 17 tags universais
que funcionam para qualquer idioma. O spaCy usa UD como padrão em `token.pos_`.

| Tag UD | PTB equiv. | Exemplo PT | Descrição |
|--------|-----------|-----------|-----------|
| NOUN | NN/NNS | cidade, modelo | Substantivo comum |
| PROPN | NNP | São Paulo, Google | Nome próprio |
| VERB | VB/VBD | correr, treinou | Verbo principal |
| AUX | MD | ser, estar, ter | Verbo auxiliar |
| ADJ | JJ | rápido, neural | Adjetivo |
| ADV | RB | muito, não | Advérbio |
| ADP | IN | de, em, para | Preposição |
| DET | DT | o, a, um | Determinante |
| PRON | PRP | eu, ele, meu | Pronome |
| CCONJ | CC | e, ou, mas | Conj. coordenativa |
| SCONJ | IN | que, se, porque | Conj. subordinativa |
| NUM | CD | dois, 42 | Numeral |
| INTJ | UH | ah, uau, opa | Interjeição |
| PUNCT | . | . , ! ? | Pontuação |

## 2.3 Token.tag_ — tags morfológicas detalhadas

Além de `token.pos_`, o spaCy fornece `token.tag_` com informação
morfológica mais granular: gênero, número, tempo, modo e pessoa.

| Token | token.pos_ | token.tag_ |
|-------|-----------|-----------|
| `'modelo'` (subj.) | NOUN | NOUN__Gender=Masc\|Number=Sing |
| `'classificou'` | VERB | VERB__Mood=Ind\|Person=3\|Tense=Past |
| `'processando'` | VERB | VERB__VerbForm=Ger |
| `'melhores'` | ADJ | ADJ__Number=Plur |
| `'os'` (art.) | DET | DET__Definite=Def\|Gender=Masc\|Number=Plur |

---

# PARTE 3 — Como Funciona um POS Tagger

## 3.1 Três gerações de taggers

### Geração 1 — Regras (1960–1990)

```
Regra: "se a palavra termina em -ção e está precedida de artigo → NOUN"
```

Funciona em domínios fechados. Frágil com variações linguísticas reais.

### Geração 2 — Estatístico (1990–2012)

Usa Modelos Ocultos de Markov (HMM) com duas probabilidades:

```
P(tag | palavra)             ← probabilidade de emissão
P(tag_atual | tag_anterior)  ← probabilidade de transição
```

O algoritmo de Viterbi encontra a sequência de tags mais provável.
Precisão: ~96% no Penn Treebank.

### Geração 3 — Neural (2012–presente)

```
BiLSTM + CRF:  ~98% de precisão
Transformers:  ~98–99% de precisão
```

O spaCy usa uma rede neural treinada para o português.

## 3.2 Ambiguidade contextual — o principal desafio

```
"O modelo neural foi treinado."  →  'modelo' = NOUN (sujeito)
"Eu modelo os dados."            →  'modelo' = VERB (1ª pessoa)

"chegou cedo"                    →  'cedo' = ADV (advérbio de tempo)
"eu cedo meu lugar"              →  'cedo' = VERB (verbo ceder)
```

O tagger usa as palavras **ao redor** para resolver a ambiguidade.

---

# PARTE 4 — POS Tagging com NLTK

## 4.1 pos_tag() em inglês

```python
from nltk.tokenize import word_tokenize
from nltk import pos_tag

frase = "The neural network quickly processed the large dataset."
tokens = word_tokenize(frase)
tags   = pos_tag(tokens)

desc = {
    'DT':'Determinante', 'JJ':'Adjetivo',   'NN':'Subs.',
    'NNS':'Subs. pl.',  'VBD':'V. passado', 'VBN':'Particípio',
    'RB':'Advérbio',    'IN':'Preposição',   'CC':'Conj.',
}

print(f'{"Token":<18} {"Tag":<8} {"Classe"}')
print('-' * 45)
for token, tag in tags:
    print(f'{token:<18} {tag:<8} {desc.get(tag, tag)}')
```

## 4.2 Ambiguidade com NLTK

```python
frases = [
    "The model achieved high accuracy.",
    "We need to model the distribution.",
]

for frase in frases:
    tokens = word_tokenize(frase)
    tags   = pos_tag(tokens)
    for token, tag in tags:
        if token.lower() == 'model':
            classe = 'SUBSTANTIVO' if tag.startswith('N') else 'VERBO'
            print(f'"{frase}"')
            print(f'  model = {tag} ({classe})')
            print()
```

---

# PARTE 5 — POS Tagging com spaCy em Português

## 5.1 Análise morfossintática completa

```python
import spacy

nlp = spacy.load('pt_core_news_sm')

texto = "A pesquisadora brasileira desenvolveu um modelo de PLN muito preciso."
doc   = nlp(texto)

print(f'{"Token":<22} {"Lema":<18} {"POS":<10} {"Tag detalhada":<30} {"Stop?"}')
print('=' * 90)

for token in doc:
    if not token.is_space:
        stop = '✓' if token.is_stop else ''
        print(f'{token.text:<22} {token.lemma_:<18} {token.pos_:<10} {token.tag_:<30} {stop}')
```

## 5.2 Filtrar por classe gramatical

```python
corpus = [
    "A inteligência artificial transforma a medicina moderna.",
    "Pesquisadores brasileiros desenvolvem algoritmos inovadores.",
    "O Brasil cresce no ranking de inovação tecnológica.",
]

doc_corpus = nlp(' '.join(corpus))

substantivos = []
verbos       = []
adjetivos    = []

for token in doc_corpus:
    if token.is_alpha and not token.is_stop and len(token.text) > 2:
        if token.pos_ in ['NOUN', 'PROPN']:
            substantivos.append(token.lemma_.lower())
        elif token.pos_ == 'VERB':
            verbos.append(token.lemma_.lower())
        elif token.pos_ == 'ADJ':
            adjetivos.append(token.lemma_.lower())

print('Substantivos:', substantivos)
print('Verbos:      ', verbos)
print('Adjetivos:   ', adjetivos)
```

---

# PARTE 6 — Visualização com displacy

## 6.1 Modo 'dep' — árvore de dependências

```python
from spacy import displacy

frase = "O modelo neural classificou os textos rapidamente."
doc   = nlp(frase)

# Tabela de dependências
print(f'{"Token":<20} {"POS":<8} {"Dep":<14} {"Head"}')
print('-' * 55)
for token in doc:
    if not token.is_punct:
        print(f'{token.text:<20} {token.pos_:<8} {token.dep_:<14} {token.head.text}')

# Renderizar no Colab
displacy.render(doc, style='dep', jupyter=True,
                options={'distance': 110, 'compact': True})
```

Rótulos de dependência mais comuns:

| Rótulo | Significado | Exemplo |
|--------|-------------|---------|
| ROOT | Raiz da oração (verbo principal) | classificou |
| nsubj | Sujeito nominal | modelo |
| obj | Objeto direto | textos |
| amod | Modificador adjetival | neural |
| advmod | Modificador adverbial | rapidamente |
| det | Determinante | o, os |

## 6.2 Modo 'ent' personalizado para POS

```python
cores_pos = {
    'NOUN' : '#2E75B6',
    'PROPN': '#1F4E79',
    'VERB' : '#1A5E3A',
    'ADJ'  : '#C45911',
    'ADV'  : '#7B61C8',
    'DET'  : '#AAAAAA',
}

frase_vis = "O pesquisador brasileiro desenvolveu um algoritmo eficiente."
doc_vis   = nlp(frase_vis)

ents = []
for token in doc_vis:
    if not token.is_space:
        ents.append({
            'start': token.idx,
            'end'  : token.idx + len(token.text),
            'label': token.pos_
        })

displacy.render(
    {'text': frase_vis, 'ents': ents, 'title': None},
    style='ent', manual=True, jupyter=True,
    options={'ents': list(cores_pos.keys()), 'colors': cores_pos}
)
```

---

# PARTE 7 — Matcher: Padrões de POS

## 7.1 Buscar padrões linguísticos

O `Matcher` do spaCy permite definir padrões de POS e buscá-los em textos.

```python
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

# Padrão: [ADJ?] NOUN [ADJ*]  ← sintagma nominal
padrao_np = [
    {'POS': 'ADJ',  'OP': '?'},   # adjetivo opcional
    {'POS': 'NOUN'},               # substantivo obrigatório
    {'POS': 'ADJ',  'OP': '*'},   # zero ou mais adjetivos
]
matcher.add('SINTAGMA_NOMINAL', [padrao_np])

# Padrão: NOUN + ADP + NOUN  ← composto nominal
padrao_comp = [
    {'POS': 'NOUN'},
    {'POS': 'ADP'},
    {'POS': {'IN': ['NOUN', 'PROPN']}},
]
matcher.add('COMPOSTO', [padrao_comp])

texto = "O sistema de processamento de linguagem natural identifica entidades nomeadas."
doc   = nlp(texto)
matches = matcher(doc)

vistos = set()
for match_id, start, end in matches:
    span = doc[start:end]
    nome = nlp.vocab.strings[match_id]
    if span.text not in vistos:
        vistos.add(span.text)
        tags = [f'{t.text}/{t.pos_}' for t in span]
        print(f'[{nome}] "{span.text}"  →  {tags}')
```

---

# PARTE 8 — Mini-ABSA com Dependências

## 8.1 Análise de Sentimento por Aspecto

ABSA (Aspect-Based Sentiment Analysis) identifica não apenas se um texto
é positivo ou negativo, mas **sobre qual aspecto** o sentimento se refere.

Estratégia usando `token.dep_` e `token.head`:
1. Encontrar adjetivos (ADJ)
2. Verificar qual substantivo eles modificam via `token.head`
3. Classificar o adjetivo como positivo ou negativo

```python
positivos = {'excelente', 'ótimo', 'rápido', 'eficiente', 'bonito', 'bom'}
negativos = {'ruim', 'péssimo', 'fraco', 'lento', 'quebrado', 'horrível'}

reviews = [
    "A câmera é excelente mas a bateria é fraca.",
    "O design é bonito e o processador é muito rápido.",
    "O preço é caro demais para um produto tão fraco.",
]

for review in reviews:
    doc_r = nlp(review)
    print(f'Review: "{review}"')
    for token in doc_r:
        if token.pos_ == 'ADJ' and token.head.pos_ in ['NOUN', 'PROPN']:
            aspecto  = token.head.lemma_
            adjetivo = token.lemma_.lower()
            if adjetivo in positivos:
                print(f'  ✅ {aspecto} → {adjetivo} (POSITIVO)')
            elif adjetivo in negativos:
                print(f'  ❌ {aspecto} → {adjetivo} (NEGATIVO)')
    print()
```

---

# Síntese da Aula 03

## Conceitos consolidados

| Conceito | Definição resumida |
|----------|--------------------|
| **Morfema** | Menor unidade com significado em uma língua |
| **POS Tagging** | Atribuir classe gramatical a cada token |
| **Penn Treebank** | Tagset de 48 tags para inglês — padrão do NLTK |
| **Universal Dependencies** | 17 tags universais cross-linguísticas — padrão do spaCy |
| **token.pos_** | Classe gramatical UD (NOUN, VERB, ADJ...) |
| **token.tag_** | Tag morfológica detalhada (gênero, número, tempo...) |
| **token.dep_** | Rótulo da relação de dependência sintática |
| **token.head** | Token governador na estrutura de dependências |
| **displacy** | Ferramenta de visualização do spaCy |
| **Matcher** | Busca padrões de POS em textos |
| **ABSA** | Sentimento por aspecto usando POS e dependências |

## Aplicações do POS Tagging

| Aplicação | Como usa POS |
|-----------|-------------|
| Lematização precisa | `'cedo'` ADV → lema `'cedo'` / `'cedo'` VERB → lema `'ceder'` |
| ABSA | ADJ + head NOUN = par (aspecto, sentimento) |
| Extração de keywords | Manter só NOUN, VERB, ADJ — content-word filtering |
| NER | PROPN é forte indício de entidade nomeada |
| Chunking | `[DET?][ADJ*][NOUN]` = sintagma nominal |

## Conexão com as próximas aulas

- **Semana 04** — BoW e TF-IDF: usar tokens filtrados por POS como features
- **Semana 09** — Parsing e Chunking: aprofundar relações sintáticas

---

## Referências Bibliográficas

| Código | Referência |
|--------|-----------|
| **B1** | CASELI, H. M.; NUNES, M. G. V. (org.). *Processamento de linguagem natural*. São Carlos: BPLN, 2023. **Cap. 4–5, pp. 85–148.** |
| **C1** | MARTINS, J. S. et al. *Processamento de Linguagem Natural*. Porto Alegre: Grupo A, 2024. **Cap. 3, pp. 49–72.** |

### Recursos online

- NLTK Book Cap. 5: [nltk.org/book/ch05.html](https://www.nltk.org/book/ch05.html)
- spaCy Linguistic Features: [spacy.io/usage/linguistic-features](https://spacy.io/usage/linguistic-features)
- Universal Dependencies: [universaldependencies.org](https://universaldependencies.org)

---

*IBM3130 · PLN · Aula 03 · 2º Semestre de 2026*
