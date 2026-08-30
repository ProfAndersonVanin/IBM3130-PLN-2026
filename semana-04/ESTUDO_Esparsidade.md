# Esparsidade em Representações Vetoriais
## Guia de Estudo — IBM3130 PLN

---

**Objetivo:** Entender o conceito de esparsidade, como calculá-la e por que é importante em Processamento de Linguagem Natural.

---

# PARTE 1 — O que é Esparsidade?

## 1.1 Definição

**Esparsidade** é a medida do percentual de valores **zero** em uma matriz ou vetor.

Em outras palavras:

$$\text{Esparsidade} = \frac{\text{número de zeros}}{\text{número total de células}} \times 100\%$$

Alternativamente, pode-se falar em **densidade** (oposto de esparsidade):

$$\text{Densidade} = \frac{\text{número de não-zeros}}{\text{número total de células}} \times 100\%$$

Relação:
$$\text{Esparsidade} + \text{Densidade} = 100\%$$

---

## 1.2 Por que uma matriz tem muitos zeros?

### Contexto 1 — Bag of Words com vocabulário grande

Quando representamos documentos usando Bag of Words (BoW), cada documento usa apenas uma **fração minúscula** do vocabulário total.

**Exemplo:**
```
Vocabulário total do corpus: 50.000 palavras diferentes
Documento específico: 300 palavras usadas
Cobertura do documento: 300 / 50.000 = 0.6%
Consequência: 99.4% das dimensões serão zero
```

### Contexto 2 — Lei de Zipf

A distribuição de frequências segue a **Lei de Zipf**:
- Algumas poucas palavras são **muito frequentes** (ex: "o", "a", "de")
- A maioria das palavras é **rara** (aparecem 1-2 vezes)

Resultado: Um documento típico toca apenas palavras comuns e algumas raras — a maioria do vocabulário não aparece.

---

# PARTE 2 — Exemplo Prático de Cálculo

## Exemplo 1 — Corpus Simples (Similar à Questão 5)

### Dados do Corpus

Suponha um corpus de avaliações de filmes:

```
Documento 1: "filme excelente, atores incríveis, recomendo"
Documento 2: "filme ruim, chato, não recomendo"
Documento 3: "filme bom, interessante, adorei"
Documento 4: "filme péssimo, horrible, não vale"
```

### Vocabulário Extraído

Após pré-processamento (tokenização, lowercase, sem stopwords):

```
Vocabulário: ["filme", "excelente", "atores", "incríveis", "recomendo", 
              "ruim", "chato", "bom", "interessante", "adorei", 
              "péssimo", "horrible", "vale"]

Total de palavras: 13
```

### Construindo a Matriz Bag of Words

| Documento | filme | excelente | atores | incríveis | recomendo | ruim | chato | bom | interessante | adorei | péssimo | horrible | vale |
|-----------|-------|-----------|--------|-----------|-----------|------|-------|-----|--------------|--------|---------|----------|------|
| Doc 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Doc 2 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Doc 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| Doc 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |

### Cálculo de Esparsidade

**Passo 1: Contar células totais**

$$\text{Total de células} = \text{documentos} \times \text{vocabulário} = 4 \times 13 = 52$$

**Passo 2: Contar células com valor zero**

Linha por linha:
- Doc 1: 8 zeros (não tem: ruim, chato, bom, interessante, adorei, péssimo, horrible, vale)
- Doc 2: 11 zeros (não tem: excelente, atores, incríveis, bom, interessante, adorei, péssimo, horrible, vale, e mais)
- Doc 3: 10 zeros
- Doc 4: 9 zeros

Total de zeros: 8 + 11 + 10 + 9 = **38 zeros**

**Passo 3: Calcular esparsidade**

$$\text{Esparsidade} = \frac{38}{52} \times 100 = 73,08\%$$

**Passo 4: Verificação com densidade**

$$\text{Não-zeros} = 52 - 38 = 14$$

$$\text{Densidade} = \frac{14}{52} \times 100 = 26,92\%$$

$$\text{Verificação:} \quad 73,08\% + 26,92\% = 100\% \quad ✓$$

---

## 1.3 Interpretação do Resultado

**Resultado:** 73,08% de esparsidade

**O que significa:**
- Quase 3/4 da matriz são valores zero
- Apenas ~27% contém informação real (valores ≠ 0)
- Para armazenar 52 valores, 38 são desperdício de espaço

---

# PARTE 3 — Exemplo 2 — Corpus Maior (Mais Realista)

## Dados

Suponha um corpus de **100 avaliações** com vocabulário de **5.000 palavras**:

### Dimensões

- Documentos: 100
- Vocabulário: 5.000
- Total de células: 100 × 5.000 = **500.000**

### Análise por documento

Cada avaliação típica contém:
- ~200 palavras diferentes
- O vocabulário tem 5.000 palavras
- Cobertura: 200 / 5.000 = 4%
- Falta de cobertura: 96%

**Número de zeros por documento:** 5.000 - 200 = 4.800 zeros

**Total de zeros no corpus:** 100 × 4.800 = 480.000

### Cálculo

$$\text{Esparsidade} = \frac{480.000}{500.000} \times 100 = 96\%$$

**Interpretação:** 96% da matriz são zeros — apenas 4% contém dados úteis!

---

# PARTE 4 — Exemplo 3 — Corpus Real (Industrial)

## Cenário: Sistema de Análise de Reviews em Ecommerce

### Dimensões

- **Documentos:** 1.000.000 de reviews
- **Vocabulário:** 100.000 palavras diferentes
- **Matriz BoW total:** 1.000.000 × 100.000 = **100 bilhões de células**

### Análise

Review típica:
- Tamanho: 50-300 palavras
- Média: ~150 palavras
- Cobertura: 150 / 100.000 = 0,15%
- Falta: 99,85%

### Cálculo

$$\text{Número de zeros por review} = 100.000 - 150 = 99.850$$

$$\text{Total de zeros} = 1.000.000 \times 99.850 = 99.850.000.000$$

$$\text{Esparsidade} = \frac{99.850.000.000}{100.000.000.000} \times 100 = 99,85\%$$

### Impacto Prático

**Armazenamento:**
- Matriz densa: 100 bilhões de números × 8 bytes = **800 GB de RAM** (impraticável!)
- Matriz esparsa: apenas 150 bilhões de valores = **~1.2 GB** (factível!)

**Computação:**
- Operações com 99,85% de zeros são desperdiçadas
- Algoritmos de matriz esparsa pulam zeros automaticamente
- Speedup: ~500–1000x mais rápido

---

# PARTE 5 — Por que Esparsidade é um Problema?

## 5.1 Problema 1: Memória

```python
import numpy as np

# Matriz densa (ruim)
matriz_densa = np.zeros((1000000, 100000))  # Aloca 100 bilhões de elementos
print(f"Tamanho em memória: {matriz_densa.nbytes / 1e9:.1f} GB")
# Output: 800.0 GB (impossível!)

# Matriz esparsa (bom)
from scipy.sparse import csr_matrix
matriz_esparsa = csr_matrix((1000000, 100000))
print(f"Tamanho em memória: {matriz_esparsa.data.nbytes / 1e6:.1f} MB")
# Output: ~1200 MB (factível!)
```

## 5.2 Problema 2: Computação

```python
# Exemplo: Multiplicação de matrizes

# Matriz densa: calcula 100 bilhões de operações
# Resultado: cada célula zero × qualquer número = zero (desperdiçado)

# Matriz esparsa: só calcula 150 bilhões de operações (onde há dados)
# Resultado: muito mais rápido
```

## 5.3 Problema 3: Aprendizado de Modelos

- **Overfitting:** Com 99,85% de zeros, o modelo pode memorizar padrões de zero
- **Signal-to-noise baixo:** Difícil distinguir padrão real de ruído
- **Convergência lenta:** Descida de gradiente é ineficiente

---

# PARTE 6 — Soluções para Reduzir Esparsidade

## 6.1 Alternativa 1: TF-IDF (Ponderação)

BoW comum trata todas as palavras igualmente. TF-IDF pondera:

$$\text{TF-IDF}(t, d) = \frac{\text{contagem}(t, d)}{|d|} \times \log\left(\frac{N}{\text{df}(t)}\right)$$

**Benefício:** Palavras raras ganham peso maior, comuns ganham peso menor → matriz resultante contém valores mais significativos (menos zeros aparentes).

## 6.2 Alternativa 2: Word Embeddings (Word2Vec, GloVe)

Em vez de criar vetor com 100.000 dimensões:
- Aprender embedding denso com **300 dimensões**
- Cada dimensão representa **conceito** (não palavra específica)
- **Resultado:** 300 números densos vs. 100.000 zeros/uns

```python
# BoW (esparso)
vetor_bow = [0, 1, 0, 0, 0, ..., 0, 1, 0, ...]  # 100.000 dims, 99.8% zeros

# Embedding (denso)
vetor_embedding = [0.82, -0.11, 0.45, 0.56, -0.03, ..., 0.19]  # 300 dims, 0% zeros
```

## 6.3 Alternativa 3: Feature Selection

Manter apenas as palavras mais informativas:
- Remover palavras que aparecem em <5 documentos
- Remover palavras que aparecem em >95% dos documentos
- Resultado: vocabulário 10.000 → 5.000 palavras

---

# PARTE 7 — Exercício Prático

## Exercício 1: Calcular Esparsidade

Dado o corpus:

```
Documento 1: "código Python funciona bem"
Documento 2: "código Java rápido"
Documento 3: "Python e Java"
```

Vocabulário após processamento: `["código", "Python", "funciona", "bem", "Java", "rápido"]`

**Tarefa:**
1. Construir a matriz BoW
2. Contar total de células
3. Contar células com zero
4. Calcular esparsidade
5. Calcular densidade
6. Verificar: esparsidade + densidade = 100%?

---

## Exercício 2: Impacto de Aumento do Corpus

Suponha um corpus que cresce:

| Tamanho | Docs | Vocab | Total células | Média palavras/doc | Zeros esperados | Esparsidade |
|---------|------|-------|---------------|--------------------|-----------------|-------------|
| Pequeno | 10 | 100 | 1.000 | 20 | 800 | **80%** |
| Médio | 100 | 1.000 | 100.000 | 50 | 95.000 | **95%** |
| Grande | 1.000 | 10.000 | 10.000.000 | 100 | 9.900.000 | **99%** |
| Enorme | 100.000 | 100.000 | 10^10 | 150 | 9.985 × 10^9 | **99,85%** |

**Observação:** Conforme o corpus cresce, **esparsidade aumenta**.

---

# PARTE 8 — Resumo e Conceitos-Chave

| Conceito | Significado | Exemplo |
|----------|------------|---------|
| **Esparsidade** | % de zeros na matriz | 96% = matriz muito esparsa |
| **Densidade** | % de não-zeros na matriz | 4% = matriz pouco densa |
| **Matriz densa** | Poucos zeros; muitos valores significativos | Embeddings (300 dims) |
| **Matriz esparsa** | Muitos zeros; poucos valores significativos | BoW (100.000 dims) |
| **Overhead de memória** | Desperdício ao armazenar zeros | 800 GB para 100k docs × 100k vocab |
| **Eficiência computacional** | Algoritmos esparsos pulam zeros | ~500-1000x mais rápido |
| **Lei de Zipf** | Poucas palavras frequentes, maioria rara | "o" > "modelo" > "implementação" |

---

# PARTE 9 — Referências nas Aulas

| Aula | Tópico | Referência |
|------|--------|-----------|
| **Aula 04** | Bag of Words e esparsidade | Seção 2.4: Limitações do BoW |
| **Aula 04** | TF-IDF como solução | Seção 3: TF-IDF |
| **Aula 06** | Word Embeddings como solução | Seção 1: Problema das Representações Esparsas |
| **Aula 02** | Stopwords reduzem esparsidade | Seção 4: Stopwords |

---

# GABARITO DO EXERCÍCIO 1

## Solução

### Matriz BoW

| Documento | código | Python | funciona | bem | Java | rápido |
|-----------|--------|--------|----------|-----|------|--------|
| Doc 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| Doc 2 | 1 | 0 | 0 | 0 | 1 | 1 |
| Doc 3 | 0 | 1 | 0 | 0 | 1 | 0 |

### Cálculo

**Total de células:** 3 × 6 = 18

**Contagem de zeros:**
- Doc 1: 2 zeros
- Doc 2: 3 zeros
- Doc 3: 4 zeros
- **Total: 9 zeros**

**Esparsidade:** 9 / 18 = 0,5 = **50%**

**Densidade:** 9 / 18 = 0,5 = **50%**

**Verificação:** 50% + 50% = 100% ✓

---

*Guia preparado para IBM3130 — PLN · Semanas 1-6 · 2026*

