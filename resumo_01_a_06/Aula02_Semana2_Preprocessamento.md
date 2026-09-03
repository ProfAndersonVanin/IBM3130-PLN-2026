# IBM3130 — Processamento de Linguagem Natural
## Aula 02 · Semana 2 · Pré-processamento de Texto

---

| | |
|---|---|
| **Disciplina** | IBM3130 — Processamento de Linguagem Natural |
| **Aula** | 02 de 14 · Semana 2 · 2º Semestre de 2026 |
| **Módulo** | Módulo 1 — Fundamentos |
| **Pré-requisito** | Aula 01 — Introdução ao PLN |
| **Ferramentas** | Google Colab · Python · NLTK · spaCy |
| **Referências** | B1 Cap. 3 (pp. 53–84) · C1 Cap. 2 · C4 Cap. 5 |

---

## Contextualização

Na Aula 01 aprendemos o que é PLN e como a linguagem humana é estruturada.
Agora chegou o momento de trabalhar com texto de verdade — e texto real é
cheio de ruído: maiúsculas, pontuação, URLs, emojis, stopwords e variações
morfológicas que precisam ser tratadas antes de qualquer análise.

Esse processo de preparação se chama **pré-processamento de texto**,
e é executado como um **pipeline** — uma sequência ordenada de etapas
onde a saída de cada uma alimenta a próxima.

> A qualidade do pré-processamento determina diretamente a qualidade
> de qualquer modelo de PLN que vier depois.

---

# PARTE 1 — O Pipeline de Pré-processamento

## 1.1 O que é um pipeline?

Um pipeline de pré-processamento é uma sequência ordenada de transformações
aplicadas a textos brutos para produzir representações limpas e padronizadas.

```
Texto bruto
   ↓  1. Tokenização
   ↓  2. Lowercase
   ↓  3. Limpeza de ruído
   ↓  4. Remoção de stopwords
   ↓  5. Stemming ou Lematização
Texto processado ✓
```

## 1.2 Por que a ordem das etapas importa

A ordem não é arbitrária — cada etapa depende das anteriores.

| Erro de ordem | Consequência |
|---------------|-------------|
| Stemming antes de remover stopwords | `'de'` pode virar `'d'` — não é mais reconhecida como stopword |
| Lowercase antes de tokenizar com NLTK | Perde pistas de maiúscula para detectar abreviações (`Dr.`) |
| Remover pontuação antes de tokenizar | `'Dr. Silva'` vira `'Dr Silva'` — tokenizador pode errar |

## 1.3 Exemplo de transformação passo a passo

```
Texto: "O produto CHEGOU QUEBRADO!! Não recomendo. Acesse: reclamacao.site/123"

1. Tokenização:    ['O','produto','CHEGOU','QUEBRADO','!!','Não','recomendo','.','Acesse',':','reclamacao.site/123']
2. Lowercase:      ['o','produto','chegou','quebrado','!!','não','recomendo','.','acesse',':','reclamacao.site/123']
3. Limpeza:        ['produto','chegou','quebrado','não','recomendo']
4. Sem stopwords:  ['produto','chegou','quebrado','não','recomendo']
5. Stemming:       ['produt','cheg','quebr','não','recomend']
```

---

# PARTE 2 — Tokenização

## 2.1 Definição

Tokenizar é dividir um texto contínuo em unidades menores chamadas **tokens**.
Um token pode ser uma palavra, um sinal de pontuação, um número ou um emoji.

## 2.2 As quatro estratégias

### Estratégia 1 — split() (ingênuo)

```python
frase = "PLN é incrível! Muito desafiador."
tokens = frase.split()
print(tokens)
# ['PLN', 'é', 'incrível!', 'Muito', 'desafiador.']
# Problema: pontuação grudada nas palavras
```

### Estratégia 2 — Expressão regular

```python
import re
tokens = re.findall(r'[a-záéíóúâêîôûãõàç]+', frase.lower())
print(tokens)
# ['pln', 'é', 'incrível', 'muito', 'desafiador']
# Melhor, mas pode falhar com abreviações e números
```

### Estratégia 3 — NLTK word_tokenize (recomendada)

```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(frase, language='portuguese')
print(tokens)
# ['PLN', 'é', 'incrível', '!', 'Muito', 'desafiador', '.']
# Separa pontuação corretamente. Use sempre language='portuguese'
```

### Estratégia 4 — Subpalavras (BPE/WordPiece)

Usada por modelos como BERT e GPT. Divide palavras raras em partes menores:

```
'processamento' → ['process', '##amento']
'incrível'      → ['incr', '##ível']
```

Garante que qualquer palavra possa ser representada — sem tokens desconhecidos.
Veremos na Semana 11 (Transformers).

## 2.3 Tokenização de sentenças

```python
from nltk.tokenize import sent_tokenize

texto = "O Dr. Silva prescreveu 500mg. A Dra. Ana confirmou o diagnóstico."

sentencas = sent_tokenize(texto, language='portuguese')
for i in range(len(sentencas)):
    print(f'[{i+1}] {sentencas[i]}')

# [1] O Dr. Silva prescreveu 500mg.
# [2] A Dra. Ana confirmou o diagnóstico.
# O NLTK reconhece 'Dr.' e 'Dra.' como abreviações — não como fim de sentença
```

---

# PARTE 3 — Normalização

## 3.1 Lowercase

Converte todos os tokens para minúsculas, eliminando variações de capitalização.

```python
tokens = ['O', 'Produto', 'É', 'BOM']

tokens_lower = []
for token in tokens:
    tokens_lower.append(token.lower())

print(tokens_lower)
# ['o', 'produto', 'é', 'bom']
```

### Quando NÃO aplicar lowercase

| Situação | Exemplo | Por quê preservar |
|----------|---------|-------------------|
| NER | `'Apple'` vs `'apple'` | Maiúscula distingue empresa de fruta |
| Sentimento intenso | `'HORRÍVEL!!!'` | Maiúsculas indicam intensidade |
| Siglas | `'PLN'`, `'DNA'` | Siglas em minúsculas viram palavras comuns |

## 3.2 Remoção de ruído

```python
import re

texto = "URGENTE! Acesse https://site.com — contato: email@empresa.com Tel: (11) 9999-0000 😍"

# Remover URLs
texto = re.sub(r'https?://\S+', '', texto)

# Remover e-mails
texto = re.sub(r'\S+@\S+', '', texto)

# Remover telefones
texto = re.sub(r'\(?\d{2}\)?[\s.-]?\d{4,5}[.-]?\d{4}', '', texto)

# Remover caracteres não-alfabéticos do português
texto = re.sub(r'[^a-záéíóúâêîôûãõàç\s]', ' ', texto.lower())

# Normalizar espaços
texto = re.sub(r'\s+', ' ', texto).strip()

print(texto)
# 'urgente acesse contato tel'
```

---

# PARTE 4 — Stopwords

## 4.1 O que são stopwords?

Stopwords são palavras de **alta frequência** e **baixo valor discriminativo**.
Aparecem em praticamente todos os textos, independentemente do tema.

```
Artigos:      o, a, os, as, um, uma
Preposições:  de, em, para, com, por
Conjunções:   e, ou, mas, porque, que
Pronomes:     eu, ele, meu, este, isso
Verbos aux.:  ser, estar, ter, haver
```

O NLTK tem uma lista de **207 stopwords** em português.

## 4.2 Removendo stopwords com NLTK

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

texto = "O sistema de PLN analisa sentimentos em textos modernos"
tokens = word_tokenize(texto.lower(), language='portuguese')

stop_pt = stopwords.words('portuguese')

tokens_sem_stop = []
for token in tokens:
    if token not in stop_pt and token.isalpha() and len(token) > 2:
        tokens_sem_stop.append(token)

print('Com stopwords:    ', tokens)
print('Sem stopwords:    ', tokens_sem_stop)
```

## 4.3 O erro mais crítico — remover 'não'

```python
# A palavra 'não' está na lista padrão do NLTK
stop_pt = stopwords.words('portuguese')
print("'não' está na lista?", 'não' in stop_pt)  # True

# O problema:
frase = "não gostei do produto"
tokens = word_tokenize(frase.lower(), language='portuguese')

tokens_sem_stop = [t for t in tokens if t not in stop_pt and t.isalpha()]
print(tokens_sem_stop)
# ['gostei', 'produto']  ← sentimento INVERTIDO!

# A solução: remover 'não' da lista de stopwords
stop_pt.remove('não')
tokens_corretos = [t for t in tokens if t not in stop_pt and t.isalpha()]
print(tokens_corretos)
# ['não', 'gostei', 'produto']  ← sentimento preservado ✓
```

## 4.4 Quando NÃO remover stopwords

| Tarefa | Motivo |
|--------|--------|
| Tradução automática | Artigos e preposições são essenciais na gramática do idioma alvo |
| Modelos BERT/GPT | Self-attention precisa do contexto completo |
| QA — perguntas e respostas | A estrutura da pergunta define o tipo de resposta |
| Análise de sentimento | `'não'`, `'nunca'`, `'muito'` carregam sentimento |

---

# PARTE 5 — Stemming

## 5.1 O que é stemming?

Stemming reduz cada palavra ao seu **radical** — a raiz morfológica aproximada —
por meio de regras heurísticas de remoção de sufixos.

```
'correndo'   → 'corr'
'correu'     → 'corr'
'corrida'    → 'corr'
'corredor'   → 'corr'
```

Todas produzem o mesmo radical — podem ser comparadas diretamente.

## 5.2 RSLPStemmer — o stemmer para o português

O RSLP (Removedor de Sufixos da Língua Portuguesa) foi desenvolvido em 2001
na UFRGS. É o stemmer padrão para o português no NLTK.

```python
from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()

palavras = ['correndo', 'correu', 'corrida', 'universidade',
            'universitário', 'computação', 'computador', 'felizmente']

print('Stemming com RSLPStemmer:')
for palavra in palavras:
    radical = stemmer.stem(palavra)
    print(f'  {palavra:<20} → {radical}')
```

Saída:
```
  correndo             → corr
  correu               → corr
  corrida              → corr
  universidade         → univers
  universitário        → univers
  computação           → comput
  computador           → comput
  felizmente           → felizm
```

## 5.3 Problemas do stemming

### Overstemming — corta demais

```
'universidade' → 'univers'
'universo'     → 'univers'
→ Palavras não relacionadas agrupadas!
```

### Understemming — não corta o suficiente

```
'melhor'  → 'melh'
'bom'     → 'bom'
→ Relacionadas semanticamente, mas radicais diferentes!
```

---

# PARTE 6 — Lematização

## 6.1 O que é lematização?

Lematização identifica a **forma canônica** (lema) de cada palavra usando
um dicionário morfológico e análise gramatical. O resultado é sempre
uma palavra válida do idioma.

```
'correndo'   → lema: 'correr'      (infinitivo)
'melhores'   → lema: 'bom'         (superlativo irregular!)
'crianças'   → lema: 'criança'     (singular)
'fui'        → lema: 'ir'          (verbo irregular)
```

## 6.2 Lematização com spaCy

```python
import spacy

nlp = spacy.load('pt_core_news_sm')

frase = "Os melhores pesquisadores estavam desenvolvendo algoritmos precisos."

doc = nlp(frase)

print(f'{"Token":<20} {"Lema":<20} {"POS":<8} {"Stopword?"}')
print('-' * 60)

for token in doc:
    if not token.is_punct and not token.is_space:
        stop = '✓' if token.is_stop else ''
        print(f'{token.text:<20} {token.lemma_:<20} {token.pos_:<8} {stop}')
```

Saída:
```
Token                Lema                 POS      Stopword?
------------------------------------------------------------
Os                   o                    DET      ✓
melhores             bom                  ADJ
pesquisadores        pesquisador          NOUN
estavam              estar                AUX      ✓
desenvolvendo        desenvolver          VERB
algoritmos           algoritmo            NOUN
precisos             preciso              ADJ
```

## 6.3 Stemming vs. Lematização

| Critério | Stemming (RSLP) | Lematização (spaCy) |
|----------|-----------------|---------------------|
| Resultado | Radical (pode ser inválido) | Lema (palavra válida) |
| Precisão | Menor | Maior |
| Velocidade | Muito rápido | Mais lento |
| Irregulares | `'melhores'` → `'melh'` | `'melhores'` → `'bom'` ✓ |
| Uso ideal | BoW, TF-IDF, protótipos | Embeddings, análise semântica |

---

# PARTE 7 — spaCy: Pipeline Completo

## 7.1 O que o spaCy entrega em uma única chamada

Ao chamar `nlp(texto)`, o spaCy executa automaticamente:

```
tokenizer → morphologizer → parser → ner → lemmatizer
```

Cada componente enriquece o objeto `doc` com atributos que ficam disponíveis
para uso imediato.

## 7.2 Atributos essenciais do token

| Atributo | O que retorna | Exemplo para 'correndo' |
|----------|---------------|------------------------|
| `token.text` | Texto original | `'correndo'` |
| `token.lower_` | Minúsculas | `'correndo'` |
| `token.lemma_` | Lema | `'correr'` |
| `token.pos_` | Classe gramatical (UD) | `'VERB'` |
| `token.tag_` | Tag morfológica detalhada | `'VERB__VerbForm=Ger'` |
| `token.is_stop` | É stopword? | `False` |
| `token.is_alpha` | Só letras? | `True` |
| `token.is_punct` | É pontuação? | `False` |
| `token.dep_` | Relação sintática | `'advcl'` |
| `token.head` | Token governador | verbo principal |

## 7.3 Pipeline completo com spaCy

```python
import spacy
from nltk.corpus import stopwords

nlp = spacy.load('pt_core_news_sm')

reviews = [
    "O produto chegou rápido e funciona perfeitamente!",
    "Péssimo. Não funciona e chegou quebrado.",
]

classes_conteudo = ['NOUN', 'VERB', 'ADJ', 'ADV']

for review in reviews:
    doc = nlp(review.lower())

    lemas = []
    for token in doc:
        if token.pos_ in classes_conteudo:
            if token.is_alpha and len(token.text) > 2:
                lemas.append(token.lemma_)

    print(f'Original: "{review}"')
    print(f'Lemas:     {lemas}')
    print()
```

Saída:
```
Original: "O produto chegou rápido e funciona perfeitamente!"
Lemas:     ['produto', 'chegar', 'rápido', 'funcionar', 'perfeitamente']

Original: "Péssimo. Não funciona e chegou quebrado."
Lemas:     ['péssimo', 'funcionar', 'chegar', 'quebrar']
```

---

# PARTE 8 — Análise de Frequência

## 8.1 FreqDist com NLTK

```python
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()
stop_pt = stopwords.words('portuguese')
stop_pt.remove('não')

texto = """
O processamento de linguagem natural transforma textos em dados.
PLN permite analisar sentimentos, traduzir idiomas e gerar textos.
Os modelos modernos de PLN usam redes neurais e grandes corpora.
"""

tokens = word_tokenize(texto.lower(), language='portuguese')

tokens_limpos = []
for token in tokens:
    if token.isalpha() and token not in stop_pt and len(token) > 2:
        tokens_limpos.append(stemmer.stem(token))

freq = FreqDist(tokens_limpos)

print('Palavras mais frequentes (radicais):')
for palavra, contagem in freq.most_common(8):
    barra = '█' * contagem
    print(f'  {palavra:<18} {contagem}  {barra}')
```

---

# Síntese da Aula 02

## Conceitos consolidados

| Conceito | Definição resumida |
|----------|--------------------|
| **Pipeline** | Sequência ordenada de transformações — a ordem importa |
| **Tokenização** | Dividir texto em unidades mínimas (tokens) |
| **Lowercase** | Padronizar para minúsculas — cuidado com NER e siglas |
| **Limpeza** | Remover URLs, e-mails, pontuação e ruído |
| **Stopwords** | Palavras muito comuns, pouco informativas — mas `'não'` é especial! |
| **Stemming** | Radical por regras heurísticas — rápido, pode ser impreciso |
| **Lematização** | Forma canônica via dicionário + POS — preciso, capta irregulares |
| **spaCy** | Pipeline integrado — tokeniza, classifica, lematiza em uma chamada |

## Fórmulas e regras práticas

```
# Regra de ouro da remoção de stopwords
Para análise de sentimento → remover 'não' da lista padrão

# Quando usar stemming
BoW, TF-IDF, corpus grande, velocidade é prioridade

# Quando usar lematização
Análise semântica, word embeddings, corpus pequeno, qualidade é prioridade

# Filtro básico de tokens
token.isalpha() AND token not in stopwords AND len(token) > 2
```

## Conexão com as próximas aulas

- **Semana 03** — POS Tagging: identificar classes gramaticais com NLTK e spaCy
- **Semana 04** — BoW e TF-IDF: usar tokens limpos para criar vetores numéricos

---

## Referências Bibliográficas

| Código | Referência |
|--------|-----------|
| **B1** | CASELI, H. M.; NUNES, M. G. V. (org.). *Processamento de linguagem natural*. São Carlos: BPLN, 2023. **Cap. 3, pp. 53–84.** |
| **C1** | MARTINS, J. S. et al. *Processamento de Linguagem Natural*. Porto Alegre: Grupo A, 2024. **Cap. 2, pp. 23–48.** |
| **C4** | NETTO, A.; MACIEL, F. *Python para Data Science e ML Descomplicado*. Rio de Janeiro: Alta Books, 2021. **Cap. 5, pp. 87–114.** |

---

*IBM3130 · PLN · Aula 02 · 2º Semestre de 2026*
