# IBM3130 — Processamento de Linguagem Natural
## Aula 01 · Semana 1 · Introdução ao PLN

---

| | |
|---|---|
| **Disciplina** | IBM3130 — Processamento de Linguagem Natural |
| **Aula** | 01 de 14 · Semana 1 · 2º Semestre de 2026 |
| **Módulo** | Módulo 1 — Fundamentos |
| **Ferramentas** | Google Colab · Python · NLTK |
| **Referências** | B1 Cap. 1–2 · C1 Cap. 1 · C4 Cap. 1 |

---

## Contextualização

Esta é a aula inaugural da disciplina. Antes de escrever qualquer linha de código,
precisamos entender o que é PLN, de onde ele veio, o que ele já consegue fazer
e quais são seus limites. Essa base conceitual guiará todas as aulas seguintes.

---

# PARTE 1 — Da Linguagem ao Processamento de Linguagem Natural

## 1.1 O que é Linguagem?

Linguagem é o sistema simbólico que os seres humanos desenvolveram para
representar o mundo, organizar o pensamento e compartilhar experiências.
Ela não é apenas um conjunto de palavras — é uma estrutura viva, construída
coletivamente ao longo de milênios, que carrega cultura, história, emoção e intenção.

Quando você diz "está frio aqui", não está apenas informando uma temperatura.
Pode estar pedindo para fechar a janela, reclamando de alguém, ou iniciando
uma conversa. A linguagem faz tudo isso ao mesmo tempo, de forma eficiente e
natural, sem que os falantes precisem pensar conscientemente em nenhuma dessas camadas.

Isso revela algo fundamental: linguagem não é só o que se diz — é o que se
quer dizer, o contexto em que se diz, e o que o outro entende a partir disso.

## 1.2 Linguagem Natural vs. Linguagem Formal

| Característica | Linguagem Natural | Linguagem Formal |
|----------------|-------------------|-----------------|
| **Origem** | Evolução espontânea nas comunidades humanas | Projetada com regras rígidas |
| **Ambiguidade** | Alta — aceita múltiplas interpretações | Zero — cada símbolo tem um único significado |
| **Variação** | Varia entre regiões, gerações, contextos | Fixa e universal |
| **Erros** | Tolerados e compreendidos | Causam falha imediata |
| **Exemplos** | Português, inglês, mandarim, guarani | Python, SQL, notação matemática |

A linguagem natural é rica e flexível justamente porque os humanos são capazes
de lidar com toda essa ambiguidade. O computador, por padrão, não é.

## 1.3 O que é Processamento de Linguagem Natural?

PLN é o campo que se dedica a construir sistemas computacionais capazes de
receber texto ou fala em linguagem natural, compreender seu significado no
contexto em que foi produzido, e gerar respostas ou ações coerentes com esse significado.

Para isso, o PLN combina três grandes áreas do conhecimento:

| Área | Contribuição para o PLN |
|------|------------------------|
| **Linguística** | Como a linguagem é estruturada — morfologia, sintaxe, semântica, pragmática |
| **Ciência da Computação** | Algoritmos, estruturas de dados e poder de processamento |
| **Inteligência Artificial** | Aprendizado de padrões a partir de dados sem programar cada regra |

### A intersecção das três áreas

- **Linguística + Computação** (sem IA): gramáticas formais, parsing por regras — Era 1
- **Computação + IA** (sem Linguística): modelos que aprendem padrões superficiais
- **Linguística + IA** (sem Computação): teoria rica sem implementação em escala
- **Linguística + Computação + IA**: PLN moderno — sistemas que entendem, processam e geram

---

# PARTE 2 — O Problema dos 80%

## 2.1 A estatística que justifica o PLN

Estima-se que entre **80% e 90%** de todos os dados produzidos no mundo são
**não estruturados** — e a maior parte deles está em forma de texto.

Cada e-mail enviado, mensagem de WhatsApp, artigo científico, contrato,
laudo médico, notícia, avaliação de produto, comentário em vídeo — tudo isso
é dado não estruturado.

## 2.2 Dado estruturado vs. não estruturado

| Tipo | Descrição | Exemplos |
|------|-----------|---------|
| **Estruturado** | Formato predefinido, campos com tipos conhecidos | Tabelas SQL, planilhas Excel |
| **Semiestruturado** | Alguma organização, mas conteúdo livre | JSON, XML, HTML |
| **Não estruturado** | Sem esquema fixo, significado implícito | E-mails, contratos, prontuários, tweets |

## 2.3 O abismo entre dado e informação

Ter o dado não significa ter a informação.

Um hospital pode ter 2 milhões de prontuários digitalizados. Os dados existem,
estão guardados. Mas se estiverem em texto livre, um sistema tradicional não
consegue responder: *"Quantos pacientes com mais de 60 anos apresentaram esse
conjunto de sintomas nos últimos 5 anos?"*

Esse abismo existe em praticamente todos os setores:

- **Banco:** milhões de e-mails de clientes contendo reclamações não analisadas
- **Varejo:** centenas de milhares de avaliações de produtos lidas por amostragem
- **Advocacia:** décadas de contratos que exigem horas de leitura manual para buscar precedentes

PLN é a tecnologia que fecha esse abismo.

---

# PARTE 3 — A Hierarquia da Compreensão

## 3.1 Os cinco níveis linguísticos

A compreensão da linguagem se organiza em cinco camadas progressivas.
Cada camada resolve um problema específico e entrega sua solução para a próxima.

### Nível 1 — Morfológico

Estuda a estrutura interna das palavras e suas categorias gramaticais.

```
"infelizmente" = in (negação) + feliz (raiz) + mente (modo)
"processamento" = process (raiz) + amento (sufixo nominalizador)
```

**Em PLN:** tokenização, stemming, lematização, POS tagging (Aulas 02 e 03)

### Nível 2 — Sintático

Estuda como as palavras se combinam para formar frases gramaticalmente corretas.

```
"O modelo neural classificou os textos"
  → [SN: O modelo neural] + [SV: classificou os textos]
  → nsubj: "modelo" governa "classificou"
```

**Em PLN:** parsing de dependências, chunking (Aula 09)

### Nível 3 — Semântico

Estuda o significado das palavras e das frases.

```
"banco" pode ser:
  → instituição financeira  (contexto: "sacar dinheiro")
  → assento                 (contexto: "praça")
  → margem de rio           (contexto: "pesca")
```

**Em PLN:** word embeddings, NER, análise de sentimento (Aulas 06–08)

### Nível 4 — Discursivo

Estuda como sentenças se conectam para formar textos coerentes.

```
"Maria saiu. Ela voltou mais tarde."
  → "Ela" = correferência de "Maria"
```

**Em PLN:** resolução de correferência, sumarização (Aulas 12–13)

### Nível 5 — Aplicado

Onde o conhecimento linguístico resolve problemas reais.

**Em PLN:** tradução, chatbots, busca semântica, QA, LLMs (Aulas 11–14)

## 3.2 Por que a hierarquia importa

Cada técnica que você aprenderá neste curso se encaixa em um desses níveis.
Erros nos níveis inferiores propagam para os superiores — uma tokenização ruim
afeta o POS tagger, que afeta o parser, que afeta a extração semântica.

---

# PARTE 4 — As Três Eras do PLN

## 4.1 Era 1 — Sistemas Baseados em Regras (1950–1990)

**Ideia:** se a linguagem tem gramática, e gramática é um conjunto de regras,
basta codificá-las para que o computador as siga.

```
Regra: "se a palavra termina em -ção e está precedida de artigo → NOUN"
```

**Resultados:** impressionantes em domínios fechados. Frágeis no mundo real.

**O problema:** para cada regra, há exceções. Para cada exceção tratada,
surgem novas. O custo de manutenção se torna proibitivo.

**Marco histórico:** Relatório ALPAC (1966) concluiu que a tradução automática
estava longe de ser viável — primeiro grande crise do campo.

## 4.2 Era 2 — Modelos Estatísticos (1990–2012)

**Ideia:** em vez de codificar regras, aprender padrões a partir de dados.

```
P("gato" | "o") = contagem("o gato") / contagem("o")
```

**Ferramentas:** HMM, Naive Bayes, SVM, n-gramas, TF-IDF

**Marco histórico:** Penn Treebank (1992) — 1 milhão de palavras anotadas
manualmente — viabilizou o treinamento estatístico.

**O limite:** representações esparsas, janelas de contexto curtas,
vocabulário fechado — nada de semântica profunda.

## 4.3 Era 3 — Deep Learning e Transformers (2012–presente)

**Ideia:** redes neurais profundas aprendem representações hierárquicas
automaticamente — sem que engenheiros definam cada nível manualmente.

```
2013: Word2Vec — palavras como vetores densos
2017: Transformer — "Attention Is All You Need"
2018: BERT — pré-treinamento + fine-tuning
2020: GPT-3 — 175 bilhões de parâmetros
2022: ChatGPT — 100 milhões de usuários em 2 meses
```

**O estado atual:** modelos que traduzem, resumem, codificam, explicam e
dialogam com qualidade comparável à humana — mas ainda com limitações importantes.

---

# PARTE 5 — Aplicações do PLN

## 5.1 O que o PLN já faz hoje

| Aplicação | Exemplos reais |
|-----------|---------------|
| **Tradução automática** | Google Translate, DeepL |
| **Assistentes virtuais** | Siri, Alexa, Google Assistant |
| **Correção contextual** | Grammarly, corretor do Word |
| **Busca semântica** | Google, busca do Spotify |
| **Análise de sentimento** | Monitoramento de redes sociais |
| **Sumarização** | Notícias resumidas automaticamente |
| **Reconhecimento de fala** | Transcrição de reuniões, legendas |
| **Geração de código** | GitHub Copilot, Claude Code |
| **QA — Perguntas e respostas** | ChatGPT, Claude, Gemini |
| **NER** | Extração de entidades em contratos e laudos |

## 5.2 As 12 tarefas fundamentais do PLN

```
1.  Tokenização             → dividir texto em unidades
2.  POS Tagging             → identificar classe gramatical
3.  Parsing                 → analisar estrutura sintática
4.  NER                     → reconhecer entidades nomeadas
5.  Resolução de correferência → identificar quem é quem no texto
6.  Análise de sentimento   → positivo, negativo, neutro
7.  Classificação de texto  → spam, tema, categoria
8.  Tradução automática     → de um idioma para outro
9.  Sumarização             → resumir documentos longos
10. QA                      → responder perguntas em linguagem natural
11. Geração de texto        → criar texto coerente e fluente
12. Recuperação de informação → buscar documentos relevantes
```

---

# PARTE 6 — Desafios do PLN

## 6.1 Os sete desafios principais

### 1. Ambiguidade lexical

```
"Banco do Brasil"  →  empresa
"banco da praça"   →  assento
"banco do rio"     →  margem
```

### 2. Ambiguidade sintática

```
"Vi o homem com o telescópio"
  → Eu usei o telescópio para ver o homem
  → O homem que eu vi tinha um telescópio
```

### 3. Correferência

```
"Maria falou com Ana. Ela estava nervosa."
  → Quem estava nervosa? Maria ou Ana?
```

### 4. Ironia e sarcasmo

```
"Que produto incrível! Quebrou no primeiro dia."
  → Sentimento real: negativo
  → Sentimento superficial: positivo ("incrível")
```

### 5. Variação linguística

```
"Tu tens razão"  (português europeu)
"Você tem razão" (português brasileiro)
"Cê tá certo"    (registro informal)
```

### 6. Pragmática

```
"Você pode fechar a janela?"
  → Pergunta sobre capacidade física? Não.
  → É um pedido. A pragmática é necessária para entender.
```

### 7. Línguas de baixo recurso

O PLN funciona bem para inglês, português, espanhol e mandarim.
Para os outros ~6.900 idiomas do mundo — incluindo 180 línguas
indígenas brasileiras — os recursos são escassos ou inexistentes.

Pesquisadores como os do NILC/USP e o projeto AmericasNLP trabalham
para reduzir essa desigualdade.

---

# PARTE 7 — O Desafio Ético: Línguas de Baixo Recurso

## 7.1 A desigualdade linguística no PLN

Quando o GPT-4 foi lançado, pesquisadores testaram suas capacidades em dezenas
de idiomas. O resultado revelou uma hierarquia clara:

```
Inglês      → excelente
Português   → muito bom
Mandarim    → bom
Swahili     → limitado
Yorùbá      → muito limitado
Guarani     → inexistente
Krenak      → inexistente
```

Essa diferença não é acidental — é produto de escolhas, prioridades e
estruturas de incentivo que podem ser modificadas.

## 7.2 Por que essa desigualdade existe

| Causa | Descrição |
|-------|-----------|
| **Mercado** | Sistemas para inglês servem centenas de milhões de usuários |
| **Dados** | A internet é ~60% em inglês — dados de treino são desiguais |
| **Pesquisa** | A maioria dos artigos de PLN é publicada em inglês por grupos do hemisfério norte |
| **Ciclo vicioso** | Sem dados → sem modelo → sem ferramentas → menos texto digital |

## 7.3 O que está em jogo

- **Acesso a serviços:** saúde, educação e justiça usam PLN — comunidades excluídas ficam fora
- **Preservação cultural:** cada língua é um arquivo de conhecimento único
- **Autonomia política:** quem não tem voz digital não participa das conversas que moldam o mundo

## 7.4 Iniciativas relevantes

- **NILC/USP:** recursos de PLN para o português brasileiro
- **AmericasNLP:** tradução para línguas indígenas das Américas
- **Mozilla Common Voice:** corpus de fala em múltiplos idiomas
- **Masakhane:** PLN para línguas africanas

---

# PARTE 8 — Prática no Colab

## 8.1 Configuração do ambiente

```python
import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

print('Pronto!')
```

## 8.2 Abordagem ingênua — split()

```python
frase = "O Processamento de Linguagem Natural é fascinante!"

# Tokenização ingênua com split()
tokens_split = frase.split()

print('Tokenização com split():')
print(tokens_split)
print('Total de tokens:', len(tokens_split))
```

Saída:
```
['O', 'Processamento', 'de', 'Linguagem', 'Natural', 'é', 'fascinante!']
Total de tokens: 7
```

**Problema:** `'fascinante!'` — a pontuação ficou grudada na palavra.

## 8.3 Abordagem profissional — NLTK

```python
from nltk.tokenize import word_tokenize, sent_tokenize

frase = "O Dr. Silva prescreveu 500mg. A Dra. Ana confirmou."

# Tokenização de palavras
tokens = word_tokenize(frase, language='portuguese')
print('word_tokenize:', tokens)

# Tokenização de sentenças
sentencas = sent_tokenize(frase, language='portuguese')
print('\nsent_tokenize:')
for i in range(len(sentencas)):
    print(f'  [{i+1}] {sentencas[i]}')
```

Saída:
```
word_tokenize: ['O', 'Dr.', 'Silva', 'prescreveu', '500mg', '.', 'A', 'Dra.', 'Ana', 'confirmou', '.']

sent_tokenize:
  [1] O Dr. Silva prescreveu 500mg.
  [2] A Dra. Ana confirmou.
```

O NLTK reconhece `'Dr.'` e `'Dra.'` como abreviações — não como fim de sentença.

## 8.4 Análise de frequência com FreqDist

```python
from nltk.probability import FreqDist
from nltk.corpus import stopwords

texto = """
O processamento de linguagem natural é uma área fascinante da inteligência artificial.
PLN permite que computadores compreendam e gerem linguagem humana.
As aplicações de PLN incluem tradução, análise de sentimento e sistemas de busca.
"""

# Tokenizar e converter para minúsculas
tokens = word_tokenize(texto.lower(), language='portuguese')

# Remover stopwords e pontuação
stop_pt = stopwords.words('portuguese')
tokens_limpos = []
for token in tokens:
    if token.isalpha() and token not in stop_pt:
        tokens_limpos.append(token)

# Calcular frequência
freq = FreqDist(tokens_limpos)

print('Palavras mais frequentes:')
for palavra, contagem in freq.most_common(8):
    barra = '█' * contagem
    print(f'  {palavra:<20} {contagem}  {barra}')
```

Saída:
```
Palavras mais frequentes:
  linguagem            3  ███
  pln                  2  ██
  processamento        1  █
  natural              1  █
  ...
```

---

# Síntese da Aula 01

## Conceitos consolidados

| Conceito | Definição resumida |
|----------|--------------------|
| **Linguagem** | Sistema simbólico para representar o mundo e compartilhar experiências |
| **Linguagem natural** | Surge espontaneamente nas comunidades — aceita ambiguidade |
| **PLN** | Campo que ensina computadores a processar linguagem humana |
| **Dado não estruturado** | 80% dos dados do mundo — texto livre sem esquema fixo |
| **Hierarquia da compreensão** | Morfológico → Sintático → Semântico → Discursivo → Aplicado |
| **Era 1** | Regras manuais (1950–1990) — robusto em domínios fechados |
| **Era 2** | Modelos estatísticos (1990–2012) — aprendem de dados |
| **Era 3** | Deep Learning / Transformers (2012–hoje) — representações automáticas |
| **Línguas de baixo recurso** | ~6.900 idiomas com poucos ou nenhum recurso computacional |

## Conexão com as próximas aulas

- **Semana 02** — Pré-processamento: tokenizar, normalizar, remover stopwords, stemming
- **Semana 03** — POS Tagging: identificar classes gramaticais automaticamente
- **Semana 04** — BoW e TF-IDF: transformar textos em vetores numéricos

---

## Referências Bibliográficas

| Código | Referência |
|--------|-----------|
| **B1** | CASELI, H. M.; NUNES, M. G. V. (org.). *Processamento de linguagem natural*. São Carlos: BPLN, 2023. **Cap. 1–2, pp. 1–52.** |
| **C1** | MARTINS, J. S. et al. *Processamento de Linguagem Natural*. Porto Alegre: Grupo A, 2024. **Cap. 1, pp. 1–22.** |
| **C4** | NETTO, A.; MACIEL, F. *Python para Data Science e Machine Learning Descomplicado*. Rio de Janeiro: Alta Books, 2021. **Cap. 1, pp. 1–28.** |

---

*IBM3130 · PLN · Aula 01 · 2º Semestre de 2026*
