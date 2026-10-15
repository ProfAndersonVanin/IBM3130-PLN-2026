# IBM3130 — Processamento de Linguagem Natural
## Aula 08 · Semana 8 · Análise de Sentimento

---

| | |
|---|---|
| **Disciplina** | IBM3130 — Processamento de Linguagem Natural |
| **Aula** | 08 de 14 · Semana 8 · 2º Semestre de 2026 |
| **Módulo** | Módulo 3 — Análise e Extração (início) |
| **Pré-requisito** | Aulas 01–06 — Fundamentos e Representações de Texto |
| **Ferramentas** | Python · NLTK · spaCy · scikit-learn · SentiLex |
| **Referências** | B1 Cap. 9 (pp. 249–284) · B2 Cap. 10 · C1 Cap. 5 |

---

## Contextualização

Nas aulas anteriores aprendemos a **representar textos como números**:
BoW, TF-IDF e Word Embeddings transformam palavras em vetores que
algoritmos de Machine Learning conseguem processar.

Agora chegou o momento de usar essas representações para resolver
um problema concreto e de alto valor econômico:

> *"O que as pessoas estão falando sobre meu produto, serviço ou marca?"*

Essa é a pergunta central da **Análise de Sentimento** (AS) — também
chamada de *Opinion Mining* ou *Sentiment Analysis*.

Empresas gastam bilhões anualmente tentando responder essa pergunta
manualmente. PLN automatiza esse processo em escala.

### O que você vai aprender nesta aula

| Conceito | O que é |
|----------|---------|
| **Análise de Sentimento** | Identificar automaticamente a polaridade emocional de um texto |
| **Polaridade** | Classificação em positivo, negativo ou neutro |
| **Abordagem léxica** | Usar dicionários de palavras com sentimento pré-definido |
| **Abordagem supervisionada** | Treinar um classificador com exemplos rotulados |
| **SentiLex** | Léxico de sentimento para o português (INESC-ID) |
| **ABSA** | Análise de sentimento por aspecto — "sobre o quê" é o sentimento |
| **Precisão, Recall, F1** | Métricas para avaliar a qualidade do classificador |
| **Matriz de confusão** | Tabela que mostra onde o modelo acerta e erra |

---

# PARTE 1 — O que é Análise de Sentimento?

## 1.1 Definição e motivação

Análise de Sentimento é o processo de identificar, extrair e quantificar
a **opinião** expressa em um texto — geralmente classificada como
**positiva**, **negativa** ou **neutra**.

### Aplicações reais e imediatas

| Setor | Aplicação | Impacto |
|-------|-----------|---------|
| **E-commerce** | Classificar reviews de produtos automaticamente | Reduz custo de moderação em 80% |
| **Financeiro** | Analisar notícias e prever movimento de ações | Trading algorítmico |
| **Saúde pública** | Monitorar humor da população em redes sociais | Detectar surtos de depressão |
| **Política** | Medir percepção de candidatos em tempo real | Análise eleitoral |
| **Atendimento** | Priorizar tickets negativos automaticamente | Reduz churn de clientes |
| **Produto** | Identificar funcionalidades mais elogiadas/criticadas | Guia o roadmap |

## 1.2 Granularidade da análise

A análise de sentimento pode operar em diferentes níveis de detalhe:

### Nível 1 — Documento
Classifica o documento inteiro como positivo ou negativo.
```
"O produto chegou rápido, funciona perfeitamente e o preço é justo."
→ POSITIVO
```

### Nível 2 — Sentença
Classifica cada sentença individualmente.
```
"A câmera é excelente."  → POSITIVO
"Mas a bateria é fraca." → NEGATIVO
```

### Nível 3 — Aspecto (ABSA)
Identifica o sentimento **sobre cada aspecto específico** do produto.
```
"A câmera é excelente mas a bateria é fraca."
→ câmera: POSITIVO
→ bateria: NEGATIVO
```

O Nível 3 é o mais valioso — e o mais desafiador. Veremos na Parte 5.

## 1.3 Por que é difícil?

```
"Que produto incrível! Quebrou no primeiro dia."
→ Ironia: o texto parece positivo mas é negativo.

"Não é ruim."
→ Dupla negação: sentimento positivo com palavras negativas.

"O fone funciona, mas poderia ter cancelamento de ruído."
→ Comparativo implícito: nem positivo nem negativo — é uma sugestão.

"Chegou antes do prazo!"
→ Sem adjetivo explícito: o sentimento está no contexto (prazo).
```

Esses casos mostram por que análise de sentimento é uma das tarefas
mais estudadas e mais difíceis do PLN.

---

# PARTE 2 — Abordagem Léxica

## 2.1 O que é uma abordagem léxica?

A abordagem léxica usa um **dicionário de sentimento** — uma lista de
palavras já classificadas como positivas ou negativas — e soma os
escores de cada palavra encontrada no texto.

**Vantagem:** não precisa de dados de treino rotulados.
**Desvantagem:** não capta contexto, ironia, negação complexa ou domínios específicos.

## 2.2 SentiLex — léxico para o português

O **SentiLex-PT** foi desenvolvido pelo INESC-ID Lisboa e é o principal
recurso léxico de sentimento para o português. Contém aproximadamente
7.000 lemas com polaridade e intensidade.

Estrutura de cada entrada:
```
palavra,classe_gramatical,polaridade,intensidade,target
excelente,ADJ,1,1,HUM
péssimo,ADJ,-1,1,HUM
```

Polaridade: `1` (positivo), `-1` (negativo), `0` (neutro)
Target: `HUM` (humano), `NHUM` (não-humano), `N0` (neutro)

## 2.3 Construindo um léxico simples

Como o SentiLex requer cadastro para download, vamos construir um
léxico manual representativo para fins didáticos:

```python
# Léxico de sentimento em português — versão didática
lexico_positivo = {
    'excelente': 2, 'ótimo': 2, 'maravilhoso': 2, 'incrível': 2,
    'perfeito': 2, 'adorei': 2, 'amei': 2, 'fantástico': 2,
    'bom': 1, 'gostei': 1, 'recomendo': 1, 'rápido': 1,
    'bonito': 1, 'resistente': 1, 'qualidade': 1, 'vale': 1,
    'satisfeito': 1, 'atendimento': 1, 'entrega': 1, 'funciona': 1,
}

lexico_negativo = {
    'péssimo': -2, 'horrível': -2, 'terrível': -2, 'lixo': -2,
    'odiei': -2, 'odeio': -2, 'detestei': -2, 'defeituoso': -2,
    'ruim': -1, 'fraco': -1, 'quebrado': -1, 'demorou': -1,
    'arrependido': -1, 'decepcionado': -1, 'problema': -1, 'falha': -1,
    'frágil': -1, 'caro': -1, 'devolveu': -1, 'parou': -1,
}

# Amplificadores e negadores
amplificadores = {'muito': 1.5, 'extremamente': 2.0, 'super': 1.5, 'bastante': 1.3}
negadores      = {'não', 'nunca', 'jamais', 'nem', 'nenhum'}

print(f'Léxico positivo: {len(lexico_positivo)} palavras')
print(f'Léxico negativo: {len(lexico_negativo)} palavras')
```

## 2.4 Classificador léxico com tratamento de negação

```python
from nltk.tokenize import word_tokenize

def classificar_lexico(texto, lexico_pos, lexico_neg, amplifs, negadores):

    tokens = word_tokenize(texto.lower(), language='portuguese')
    escore = 0
    negacao_ativa = False

    for i in range(len(tokens)):
        token = tokens[i]

        # Verificar se é negador
        if token in negadores:
            negacao_ativa = True
            continue

        # Calcular amplificador (palavra anterior)
        fator = 1.0
        if i > 0 and tokens[i-1] in amplifs:
            fator = amplifs[tokens[i-1]]

        # Calcular escore da palavra
        if token in lexico_pos:
            contrib = lexico_pos[token] * fator
            escore  = escore + (-contrib if negacao_ativa else contrib)
            negacao_ativa = False

        elif token in lexico_neg:
            contrib = abs(lexico_neg[token]) * fator
            escore  = escore + (contrib if negacao_ativa else -contrib)
            negacao_ativa = False

        else:
            # Palavra neutra não desativa a negação imediatamente
            if token not in ['o', 'a', 'os', 'as', 'um', 'uma', 'de', 'da', 'do']:
                negacao_ativa = False

    # Classificar
    if escore > 0:
        return 'POSITIVO', escore
    elif escore < 0:
        return 'NEGATIVO', escore
    else:
        return 'NEUTRO', escore

# Testar
reviews_teste = [
    "O produto é excelente e chegou muito rápido!",
    "Péssima qualidade. Quebrou no primeiro dia.",
    "Não gostei do produto. Muito fraco para o preço.",
    "Não é ruim, mas poderia ser melhor.",
    "Produto ok. Entrega normal.",
]

print('=== Classificação Léxica ===\n')
for review in reviews_teste:
    classe, escore = classificar_lexico(
        review, lexico_positivo, lexico_negativo,
        amplificadores, negadores
    )
    print(f'  Review: "{review}"')
    print(f'  → {classe} (escore: {escore:+.1f})')
    print()
```

## 2.5 Limitações da abordagem léxica

```python
# Casos onde o léxico falha
casos_dificeis = [
    ("Que produto incrível! Quebrou no primeiro dia.",
     "Ironia — 'incrível' soa positivo mas o contexto é negativo"),
    ("Não é ruim.",
     "Dupla negação — deveria ser positivo, mas 'ruim' é negativo"),
    ("Chegou antes do prazo!",
     "Sentimento implícito — não há palavra de sentimento explícita"),
    ("Para o preço, está ótimo.",
     "Sentimento condicional — positivo apenas dado o contexto do preço"),
]

print('=== Casos onde o léxico falha ===\n')
for texto, explicacao in casos_dificeis:
    classe, escore = classificar_lexico(
        texto, lexico_positivo, lexico_negativo,
        amplificadores, negadores
    )
    print(f'  Texto:      "{texto}"')
    print(f'  Resultado:   {classe} (escore: {escore:+.1f})')
    print(f'  Problema:    {explicacao}')
    print()
```

---

# PARTE 3 — Abordagem Supervisionada

## 3.1 A ideia

Em vez de usar regras e dicionários, deixamos o modelo **aprender**
os padrões de sentimento a partir de exemplos já rotulados.

```
Dados de treino:
  "produto excelente" → POSITIVO  ┐
  "chegou quebrado"   → NEGATIVO  ├─ o modelo aprende os padrões
  "entrega rápida"    → POSITIVO  │
  "péssima qualidade" → NEGATIVO  ┘

Dados de teste:
  "atendimento horrível" → ? ← o modelo precisa prever
```

## 3.2 Pipeline supervisionado com scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# Corpus rotulado — reviews de produtos em português
reviews = [
    # Positivas (1)
    ("produto excelente chegou rápido funciona perfeitamente", 1),
    ("qualidade incrível vale cada centavo recomendo bastante", 1),
    ("entrega rápida produto bonito resistente adorei", 1),
    ("ótima compra superou expectativas muito satisfeito", 1),
    ("produto perfeito atendimento excelente voltarei comprar", 1),
    ("chegou antes prazo embalagem ótima produto funcionando", 1),
    ("bom custo benefício produto resistente recomendo", 1),
    ("excelente qualidade melhor produto comprei este ano", 1),
    ("produto fantástico superou expectativas entrega rápida", 1),
    ("muito bom produto funciona perfeitamente recomendo", 1),
    # Negativas (0)
    ("produto péssimo parou funcionar dois dias lixo", 0),
    ("horrível chegou quebrado atendimento desastre", 0),
    ("não funciona anunciado qualidade ruim muito frágil", 0),
    ("decepcionado produto defeituoso arrependido compra", 0),
    ("pior compra vida produto pessimo não recomendo", 0),
    ("produto falhou semana problema atendimento terrível", 0),
    ("qualidade horrível quebrou facilmente muito ruim", 0),
    ("não gostei produto fraco preço alto decepcionante", 0),
    ("produto defeituoso devolvi atendimento demorado ruim", 0),
    ("péssima experiência produto parou funcionar jamais compro", 0),
]

# Separar textos e rótulos
textos  = [r[0] for r in reviews]
rotulos = [r[1] for r in reviews]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    textos, rotulos,
    test_size    = 0.3,
    random_state = 42,
    stratify     = rotulos
)

print(f'Treino: {len(X_train)} reviews')
print(f'Teste:  {len(X_test)} reviews')
```

## 3.3 Vetorização com TF-IDF

```python
# Vetorizar com TF-IDF — unigramas + bigramas
vetorizador = TfidfVectorizer(
    ngram_range = (1, 2),    # unigramas e bigramas
    min_df      = 1,         # mínimo de documentos
    max_features= 500        # máximo de features
)

X_train_vec = vetorizador.fit_transform(X_train)
X_test_vec  = vetorizador.transform(X_test)

print(f'Dimensões da matriz de treino: {X_train_vec.shape}')
print(f'Dimensões da matriz de teste:  {X_test_vec.shape}')

# Ver as features mais informativas
nomes = vetorizador.get_feature_names_out()
print(f'\nAlgumas features: {list(nomes[:20])}')
```

## 3.4 Treinar e avaliar dois classificadores

```python
# Classificador 1 — Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
y_pred_nb = nb.predict(X_test_vec)

# Classificador 2 — Regressão Logística
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_vec, y_train)
y_pred_lr = lr.predict(X_test_vec)

# Resultados
for nome, y_pred in [('Naive Bayes', y_pred_nb), ('Regressão Logística', y_pred_lr)]:
    print(f'=== {nome} ===')
    print(classification_report(
        y_test, y_pred,
        target_names=['Negativo', 'Positivo'],
        zero_division=0
    ))
```

## 3.5 Matriz de confusão

```python
import matplotlib.pyplot as plt
import matplotlib

# Matriz de confusão para Regressão Logística
mc = confusion_matrix(y_test, y_pred_lr)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(mc, cmap='Blues')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Negativo', 'Positivo'], fontsize=12)
ax.set_yticklabels(['Negativo', 'Positivo'], fontsize=12)
ax.set_xlabel('Previsto', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
ax.set_title('Matriz de Confusão — Regressão Logística', fontsize=13)

# Adicionar valores nas células
for i in range(2):
    for j in range(2):
        cor  = 'white' if mc[i, j] > mc.max() / 2 else 'black'
        texto = ax.text(j, i, mc[i, j],
                        ha='center', va='center',
                        color=cor, fontsize=16, fontweight='bold')

plt.colorbar(im)
plt.tight_layout()
plt.show()

# Interpretar
print('Interpretação da matriz:')
print(f'  Verdadeiro Negativo (TN): {mc[0,0]} — negativos classificados corretamente')
print(f'  Falso Positivo     (FP): {mc[0,1]} — negativos classificados como positivo')
print(f'  Falso Negativo     (FN): {mc[1,0]} — positivos classificados como negativo')
print(f'  Verdadeiro Positivo(TP): {mc[1,1]} — positivos classificados corretamente')
```

---

# PARTE 4 — Métricas de Avaliação

## 4.1 Por que a acurácia não basta?

Imagine um corpus com 90% de reviews positivas e 10% negativas.
Um classificador que sempre prevê "positivo" teria **90% de acurácia**
— mas seria inútil para detectar reviews negativas.

Precisamos de métricas que avaliem cada classe separadamente.

## 4.2 As quatro métricas essenciais

```
                     Previsto
                  Neg    Pos
Real  Neg  │  TN  │  FP  │
      Pos  │  FN  │  TP  │
```

### Precisão
> "Dos que o modelo disse serem positivos, quantos realmente são?"

```
Precisão = TP / (TP + FP)
```

Alta precisão → poucas falsas alarmes — importante quando o custo de
um falso positivo é alto (ex.: bloquear review legítima por engano).

### Recall (Sensibilidade)
> "Dos que realmente são positivos, quantos o modelo encontrou?"

```
Recall = TP / (TP + FN)
```

Alto recall → poucos positivos perdidos — importante quando o custo de
um falso negativo é alto (ex.: não detectar uma crise de reputação).

### F1-Score
> Média harmônica entre Precisão e Recall — equilibra os dois.

```
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
```

### Acurácia
> "Percentual total de previsões corretas."

```
Acurácia = (TP + TN) / (TP + TN + FP + FN)
```

## 4.3 Calculando manualmente

```python
# Cálculo manual das métricas para o classificador de Regressão Logística
TP = mc[1, 1]
TN = mc[0, 0]
FP = mc[0, 1]
FN = mc[1, 0]

precisao = TP / (TP + FP) if (TP + FP) > 0 else 0
recall   = TP / (TP + FN) if (TP + FN) > 0 else 0
f1       = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0
acuracia = (TP + TN) / (TP + TN + FP + FN)

print('=== Métricas calculadas manualmente ===\n')
print(f'  Verdadeiro Positivo (TP): {TP}')
print(f'  Verdadeiro Negativo (TN): {TN}')
print(f'  Falso Positivo     (FP): {FP}')
print(f'  Falso Negativo     (FN): {FN}')
print()
print(f'  Precisão:  {precisao:.4f}  ({precisao*100:.1f}%)')
print(f'  Recall:    {recall:.4f}  ({recall*100:.1f}%)')
print(f'  F1-Score:  {f1:.4f}  ({f1*100:.1f}%)')
print(f'  Acurácia:  {acuracia:.4f}  ({acuracia*100:.1f}%)')
```

## 4.4 Qual métrica usar em cada situação?

| Situação | Priorizar | Por quê |
|----------|-----------|---------|
| Detectar reviews negativas de crise | **Recall** | Não pode deixar passar nenhuma |
| Filtrar spam de avaliações falsas | **Precisão** | Não pode bloquear review legítima |
| Comparar modelos de forma geral | **F1-Score** | Equilibra os dois |
| Classes perfeitamente balanceadas | **Acurácia** | Só nesse caso é suficiente |

---

# PARTE 5 — ABSA — Análise de Sentimento por Aspecto

## 5.1 Motivação

A análise de sentimento de documento diz que uma review é negativa.
Mas **o que** é negativo? O preço? A entrega? O produto em si? O atendimento?

Para uma empresa, essa distinção é crucial — cada aspecto é responsabilidade
de uma equipe diferente.

## 5.2 Abordagem com POS Tagging e dependências

Estratégia usando spaCy:
1. Encontrar **adjetivos** no texto (ADJ)
2. Identificar o **substantivo** que cada adjetivo modifica (`token.head`)
3. Classificar o adjetivo como positivo ou negativo usando o léxico

```python
import spacy

nlp = spacy.load('pt_core_news_sm')

# Léxico combinado
lexico_sent = {}
lexico_sent.update({p: 1  for p in lexico_positivo})
lexico_sent.update({p: -1 for p in lexico_negativo})

def absa(texto, lexico):
    doc      = nlp(texto)
    aspectos = []

    for token in doc:
        # Encontrar adjetivos
        if token.pos_ == 'ADJ':
            adjetivo = token.lemma_.lower()
            # O substantivo que o adjetivo modifica
            if token.head.pos_ in ['NOUN', 'PROPN']:
                aspecto = token.head.lemma_.lower()
                polaridade = lexico.get(adjetivo, 0)

                # Verificar negação (token pai tem negação?)
                for filho in token.head.children:
                    if filho.dep_ == 'advmod' and filho.lemma_.lower() in negadores:
                        polaridade = -polaridade

                if polaridade != 0:
                    sentimento = 'POSITIVO' if polaridade > 0 else 'NEGATIVO'
                    aspectos.append((aspecto, adjetivo, sentimento))

    return aspectos

# Testar ABSA
reviews_absa = [
    "A câmera é excelente mas a bateria é fraca.",
    "O design é bonito e o processador é muito rápido, mas o preço está caro.",
    "A entrega foi rápida porém o produto chegou quebrado.",
    "Atendimento péssimo mas produto bom.",
]

print('=== ABSA — Análise por Aspecto ===\n')
for review in reviews_absa:
    resultados = absa(review, lexico_sent)
    print(f'Review: "{review}"')
    if resultados:
        for aspecto, adjetivo, sentimento in resultados:
            emoji = '✅' if sentimento == 'POSITIVO' else '❌'
            print(f'  {emoji} {aspecto:<15} → {adjetivo} ({sentimento})')
    else:
        print('  (nenhum aspecto identificado)')
    print()
```

## 5.3 Limitações do ABSA baseado em regras

| Limitação | Exemplo |
|-----------|---------|
| Sentimento implícito | "Chegou antes do prazo" — sem adjetivo |
| Adjetivo sem substantivo explícito | "Rápido e eficiente!" |
| Aspecto mencionado distante do adjetivo | "O produto que comprei semana passada é ruim" |
| Ironia | "Que câmera incrível! Tirei fotos totalmente borradas." |

---

# PARTE 6 — Comparativo das Abordagens

## 6.1 Léxica vs. Supervisionada vs. ABSA

| Critério | Léxica | Supervisionada | ABSA |
|----------|--------|----------------|------|
| **Dados rotulados necessários** | Não | Sim | Não (regras) |
| **Adaptação a domínio** | Baixa | Alta | Média |
| **Trata negação** | Parcialmente | Sim (aprende) | Parcialmente |
| **Trata ironia** | Não | Parcialmente | Não |
| **Granularidade** | Documento | Documento | Aspecto |
| **Custo de implementação** | Baixo | Médio | Médio |
| **Interpretabilidade** | Alta | Baixa | Alta |

## 6.2 Como escolher a abordagem

```
Tenho dados rotulados?
  NÃO → Abordagem léxica (rápida, interpretável)
  SIM → Abordagem supervisionada (mais precisa)

Preciso saber SOBRE O QUE é o sentimento?
  SIM → ABSA (aspecto-based)
  NÃO → Documento ou sentença

Corpus é grande e diversificado?
  SIM → BERT fine-tuned (Aula 11) — melhor resultado
  NÃO → TF-IDF + Logística ou léxico
```

---

# PARTE 7 — Pipeline Completo

## 7.1 Integrando pré-processamento e classificação

```python
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()
stop_pt = stopwords.words('portuguese')
stop_pt.remove('não')   # preservar negação!

def preprocessar(texto):
    tokens = word_tokenize(texto.lower(), language='portuguese')
    tokens_limpos = []
    for token in tokens:
        if token.isalpha() and token not in stop_pt and len(token) > 2:
            tokens_limpos.append(stemmer.stem(token))
    return ' '.join(tokens_limpos)

# Aplicar pré-processamento antes de vetorizar
reviews_proc = [preprocessar(r[0]) for r in reviews]
rotulos_proc  = [r[1] for r in reviews]

print('Antes e depois do pré-processamento:')
for i in range(3):
    print(f'\n  Original:   "{reviews[i][0]}"')
    print(f'  Processado: "{reviews_proc[i]}"')

# Treinar com textos pré-processados
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    reviews_proc, rotulos_proc,
    test_size=0.3, random_state=42, stratify=rotulos_proc
)

vet2   = TfidfVectorizer(ngram_range=(1,2))
X_tr2  = vet2.fit_transform(X_train2)
X_te2  = vet2.transform(X_test2)

lr2    = LogisticRegression(random_state=42, max_iter=1000)
lr2.fit(X_tr2, y_train2)
y_p2   = lr2.predict(X_te2)

print('\n=== Pipeline Completo — Resultados ===')
print(classification_report(
    y_test2, y_p2,
    target_names=['Negativo', 'Positivo'],
    zero_division=0
))
```

## 7.2 Classificar novas reviews

```python
# Classificar reviews novas com o pipeline completo
novas_reviews = [
    "Produto chegou muito rápido e funciona perfeitamente!",
    "Péssima compra. Quebrou na primeira semana de uso.",
    "Não gostei. Qualidade muito abaixo do esperado.",
    "Bom produto pelo preço. Recomendo para quem busca custo-benefício.",
    "Entrega demorou mas o produto vale a pena.",
]

print('=== Classificando novas reviews ===\n')
for review in novas_reviews:
    proc   = preprocessar(review)
    vec    = vet2.transform([proc])
    pred   = lr2.predict(vec)[0]
    proba  = lr2.predict_proba(vec)[0]
    classe = 'POSITIVO ✅' if pred == 1 else 'NEGATIVO ❌'
    conf   = max(proba)
    print(f'  Review:     "{review}"')
    print(f'  Resultado:   {classe}  (confiança: {conf:.1%})')
    print()
```

---

# Síntese da Aula 08

## Conceitos consolidados

| Conceito | Definição resumida |
|----------|--------------------|
| **Análise de Sentimento** | Classificar a polaridade emocional de um texto |
| **Abordagem léxica** | Usar dicionário de palavras com escores pré-definidos |
| **SentiLex-PT** | Léxico de sentimento para o português — ~7.000 lemas |
| **Negação** | Inverte a polaridade — `'não gostei'` ≠ `'gostei'` |
| **Amplificador** | Intensifica a polaridade — `'muito bom'` > `'bom'` |
| **Abordagem supervisionada** | Treinar classificador com exemplos rotulados |
| **TF-IDF + Logística** | Pipeline clássico de classificação de texto |
| **ABSA** | Sentimento por aspecto — câmera POSITIVO, bateria NEGATIVO |
| **Precisão** | TP / (TP + FP) — qualidade das previsões positivas |
| **Recall** | TP / (TP + FN) — cobertura dos positivos reais |
| **F1-Score** | Média harmônica entre Precisão e Recall |
| **Matriz de confusão** | Tabela de acertos e erros por classe |

## Fórmulas essenciais

```
Precisão  = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-Score  = 2 × (P × R) / (P + R)
Acurácia  = (TP + TN) / (TP + TN + FP + FN)

Escore léxico:
  escore += polaridade(palavra) × fator_amplificador
  escore *= -1  se houver negador ativo
```

## Conexão com as próximas aulas

- **Aula 09 — NER:** reconhecer entidades como pessoas, organizações e locais — frequentemente combinado com análise de sentimento para saber "sentimento sobre quem"
- **Aula 11 — Transformers:** BERT e BERTimbau para análise de sentimento contextual — superam TF-IDF + Logística em todos os benchmarks
- **Projeto final:** análise de sentimento por aspecto em reviews de apps, produtos ou redes sociais

---

## Referências Bibliográficas

| Código | Referência |
|--------|-----------|
| **B1** | CASELI, H. M.; NUNES, M. G. V. (org.). *Processamento de linguagem natural*. São Carlos: BPLN, 2023. **Cap. 9, pp. 249–284.** |
| **B2** | FACELI, K. et al. *Inteligência Artificial: uma abordagem de aprendizado de máquina*. Rio de Janeiro: LTC, 2025. **Cap. 10.** |
| **C1** | MARTINS, J. S. et al. *Processamento de Linguagem Natural*. Porto Alegre: Grupo A, 2024. **Cap. 5, pp. 99–124.** |

### Recursos online

- **SentiLex-PT:** [github.com/aritter/twitter_nlp](https://github.com/INESC-ID/SentiLex) — léxico de sentimento para português
- **NLTK Sentiment:** [nltk.org/howto/sentiment.html](https://www.nltk.org/howto/sentiment.html)
- **scikit-learn — Naive Bayes:** [scikit-learn.org/stable/modules/naive_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)
- **BERTimbau:** [huggingface.co/neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased)

---

*IBM3130 · PLN · Aula 08 · 2º Semestre de 2026*
