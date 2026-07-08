# 📚 IBM3130 — Processamento de Linguagem Natural

**Ibmec · Ciência de Dados e Inteligência Artificial · 2º Semestre de 2026**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebooks-orange?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7%2B-09A3D5)](https://spacy.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Sobre a disciplina

Este repositório reúne todos os materiais, notebooks, atividades e recursos utilizados ao longo da disciplina **IBM3130 — Processamento de Linguagem Natural**, ofertada no segundo semestre de 2026 no curso de Ciência de Dados e Inteligência Artificial do Ibmec.

O curso cobre o campo do PLN desde seus fundamentos linguísticos e matemáticos até as arquiteturas modernas baseadas em Transformers, com foco em aplicações práticas implementadas em Python no ambiente Google Colab.

> **Nível:** Iniciantes em programação com conceitos básicos de Python  
> **Carga horária:** 14 encontros · 220 minutos cada  
> **Ambiente prático:** Google Colaboratory  
> **Professor:** Anderson Vanin

---

## Estrutura do repositório

```
IBM3130-PLN-2026/
│
├── semana-01/          # Introdução ao PLN
├── semana-02/          # Pré-processamento de texto
├── semana-03/          # Morfossintaxe e POS Tagging
├── semana-04/          # Bag of Words e TF-IDF
├── semana-05/          # Modelos N-gram e probabilísticos
├── semana-06/          # Word Embeddings
├── semana-07/          # Representações conceituais
├── semana-08/          # Reconhecimento de Entidades Nomeadas
├── semana-09/          # Sintaxe, Parsing e Chunking
├── semana-10/          # Redes neurais para PLN
├── semana-11/          # Transformers e modelos pré-treinados
├── semana-12/          # Aplicações de PLN
├── semana-13/          # Chatbots e aplicações multimodais
├── semana-14/          # Projeto final e apresentações
│
├── datasets/           # Corpora e conjuntos de dados utilizados
├── slides/             # Slides de aula em PDF
├── bibliografia/       # Referências e materiais de leitura
├── atividades/         # AC1, AC2 e gabaritos
└── README.md
```

Cada pasta de semana contém:

```
semana-XX/
├── notebook_aula.ipynb      # Notebook principal da aula
├── notebook_exercicios.ipynb # Exercícios guiados
├── material_teorico.pdf     # Material de apoio teórico
└── README.md                # Descrição dos conteúdos da semana
```

---

## Conteúdo programático

### Módulo 1 — Fundamentos e Pré-processamento `semanas 1–3`

<details>
<summary><strong>Semana 01 · Introdução ao PLN</strong></summary>

**Tópico do programa:** Tópico 1 — Introdução ao Processamento de Linguagem Natural

**Conteúdo teórico**
- O que é linguagem, linguagem natural e processamento
- Definição formal de PLN e suas fronteiras disciplinares
- A intersecção entre Linguística, Ciência da Computação e Inteligência Artificial
- Os 80% dos dados não estruturados: por que PLN importa
- A hierarquia da compreensão: níveis morfológico, sintático, semântico, discursivo e aplicado
- Evolução histórica: Era das Regras (1950–1990), Era Estatística (1990–2012), Era do Deep Learning (2012–presente)
- Aplicações do dia a dia: tradução, assistentes virtuais, correção contextual, busca semântica, LLMs
- Principais desafios: ambiguidade lexical, ambiguidade sintática, correferência, ironia, variação linguística
- O desafio ético: línguas de baixo recurso e desigualdade linguística

**Prática no Colab**
- Acesso ao Google Colab e criação do primeiro notebook
- Abordagem ingênua: tokenização com `split()`
- Abordagem profissional: `word_tokenize()` e `sent_tokenize()` com NLTK
- Análise de frequência com `FreqDist`

**Ferramentas:** `Python` · `NLTK` · `Google Colab`

**Referências:** B1 Cap. 1–2 · C1 Cap. 1 · C4 Cap. 1

</details>

<details>
<summary><strong>Semana 02 · Pré-processamento de Texto</strong></summary>

**Conteúdo teórico**
- Pipeline de PLN: do texto bruto ao dado limpo
- Tokenização avançada: estratégias por espaço, regex, modelos treinados e subpalavras
- Normalização: conversão para minúsculas, remoção de ruído (URLs, e-mails, emojis)
- Stopwords: o que são, quando remover e quando não remover
- Stemming com RSLPStemmer: redução heurística ao radical
- Lematização com spaCy: forma canônica via dicionário + POS tag
- Stemming vs. lematização: diferenças, vantagens e casos de uso

**Prática no Colab**
- Comparação de tokenizadores: `split()` vs. regex vs. NLTK vs. spaCy
- Remoção de stopwords com lista customizada do NLTK
- Aplicação do `RSLPStemmer` para o português
- Lematização com `spaCy` (modelo `pt_core_news_sm`)
- Construção de função `preprocessar()` reutilizável

**Ferramentas:** `NLTK` · `spaCy` · `re`

**Referências:** B1 Cap. 3 · C1 Cap. 2 · C4 Cap. 5

</details>

<details>
<summary><strong>Semana 03 · Marcação Morfossintática — POS Tagging</strong></summary>

**Tópico do programa:** Tópico 5 — Marcação Morfossintática

**Conteúdo teórico**
- Morfologia: morfemas, classes gramaticais e análise morfológica
- Classes gramaticais em profundidade: NOUN, PROPN, VERB, AUX, ADJ, ADV, ADP, DET, PRON, NUM
- Tagsets: Penn Treebank (48 tags) e Universal Dependencies (17 UPOS tags)
- `token.tag_` do spaCy: morfologia detalhada (gênero, número, tempo, modo, pessoa)
- Três gerações de POS taggers: regras → HMM/MaxEnt → BiLSTM+CRF → Transformers
- Aplicações: lematização, análise de sentimento por aspecto, extração de keywords, NER, chunking

**Prática no Colab**
- `nltk.pos_tag()` em inglês com Penn Treebank
- `token.pos_`, `token.tag_`, `token.dep_`, `token.head` com spaCy em português
- Visualização com `displacy` (modos `ent` e `dep`)
- Uso do `Matcher` do spaCy para busca de padrões POS
- Análise comparativa de distribuição de POS entre gêneros textuais
- Mini-ABSA: extração de aspecto-sentimento com dependências

**Ferramentas:** `NLTK` · `spaCy` · `displacy`

**Referências:** B1 Cap. 4–5 · C1 Cap. 3

</details>

---

### Módulo 2 — Representações de Texto `semanas 4–7`

<details>
<summary><strong>Semana 04 · Bag of Words e TF-IDF</strong></summary>

**Tópico do programa:** Tópico 3 — Modelos de Linguagem (parte 1)

**Conteúdo teórico**
- Representação vetorial de texto: intuição e motivação
- Bag of Words: vocabulário, contagem de frequências e matriz documento-termo
- TF-IDF: fórmula, interpretação e relevância discriminativa de termos
- Limitações: esparsidade, ausência de ordem e semântica

**Prática no Colab**
- `CountVectorizer` e `TfidfVectorizer` com scikit-learn
- Matriz documento-termo visualizada em DataFrame com pandas
- Classificador de spam com BoW + Naive Bayes
- Comparação visual BoW vs. TF-IDF em gráfico de barras

**Ferramentas:** `scikit-learn` · `pandas` · `numpy`

**Referências:** B1 Cap. 6 · B2 Cap. 5 e 8 · C3 Cap. 4

</details>

<details>
<summary><strong>Semana 05 · Modelos N-gram e Probabilísticos</strong></summary>

**Tópico do programa:** Tópico 3 — Modelos de Linguagem (parte 2)

**Conteúdo teórico**
- N-gramas: unigram, bigram e trigram
- Modelos probabilísticos de linguagem: probabilidade de emissão e transição
- Suavização de Laplace para n-gramas não observados
- Aplicações: autocomplete, correção ortográfica contextual e geração simples de texto

**Prática no Colab**
- Extração de n-gramas com NLTK e `CountVectorizer`
- Análise de frequência de bigramas em corpus real
- Geração de texto com modelo bigrama
- Visualização das palavras mais frequentes com matplotlib

**Ferramentas:** `NLTK` · `matplotlib` · `collections.Counter`

**Referências:** B1 Cap. 7 · B2 Cap. 8 · C1 Cap. 4

</details>

<details>
<summary><strong>Semana 06 · Word Embeddings — Word2Vec e GloVe</strong></summary>

**Tópico do programa:** Tópico 2 — Representações Vetoriais de Palavras

**Conteúdo teórico**
- Limitações do BoW: sem semântica nem contexto
- Intuição dos embeddings: palavras como vetores densos
- Word2Vec: arquiteturas CBOW e Skip-gram
- GloVe: representações baseadas em co-ocorrências globais
- Analogias vetoriais: `rei − homem + mulher ≈ rainha`
- Similaridade de cosseno e vizinhança semântica

**Prática no Colab**
- Embeddings pré-treinados com Gensim
- `most_similar()` e analogias vetoriais
- Visualização 2D com PCA e matplotlib
- Comparação de vizinhança semântica entre modelos

**Ferramentas:** `Gensim` · `scikit-learn (PCA)` · `matplotlib`

**Referências:** B1 Cap. 8 · B3 Cap. 2 · C3 Cap. 5

</details>

<details>
<summary><strong>Semana 07 · Representações Conceituais — WordNet e BabelNet</strong></summary>

**Tópico do programa:** Tópico 8 — Representações Conceituais de Palavras

**Conteúdo teórico**
- Ontologias e grafos de conhecimento
- WordNet: synsets, sinonímia, hiperonímia e hiponímia
- Similaridade semântica: Wu-Palmer, Lin e Path similarity
- BabelNet: base lexical multilíngue e cross-linguística
- Knowledge graphs: integração de ontologias com PLN

**Prática no Colab**
- WordNet com NLTK: `synsets()`, `definition()`, `examples()`, `hypernyms()`
- Navegação na hierarquia de hipônimos e hiperônimos
- Cálculo de similaridade semântica entre conceitos
- Visualização de grafos de conhecimento com NetworkX

**Ferramentas:** `NLTK WordNet` · `networkx`

**Referências:** B1 Cap. 9 · C1 Cap. 5

</details>

---

### Módulo 3 — Extração de Informação `semanas 8–9`

<details>
<summary><strong>Semana 08 · Reconhecimento de Entidades Nomeadas (NER)</strong></summary>

**Tópico do programa:** Tópico 4 — Extração de Informações

**Conteúdo teórico**
- Entidades nomeadas: PER, ORG, LOC, DATE, MONEY, CARDINAL
- Abordagens para NER: regras, CRF e redes neurais
- Esquema de anotação IOB/BIO: B-ENT, I-ENT, O
- Extração de relações entre entidades
- Avaliação: precisão, recall e F1 para NER

**Prática no Colab**
- NER com spaCy: `ent.text`, `ent.label_`, `ent.start_char`
- Visualização de entidades com `displacy` no Colab
- `EntityRuler`: NER personalizado com padrões de regras
- Extração de entidades em notícias reais
- Avaliação de desempenho do modelo em corpus anotado

**Ferramentas:** `spaCy` · `displacy` · `EntityRuler`

**Referências:** B1 Cap. 10–11 · C1 Cap. 6

</details>

<details>
<summary><strong>Semana 09 · Sintaxe, Parsing e Chunking</strong></summary>

**Tópicos do programa:** Tópicos 6 e 7 — Sintaxe, Semântica e Técnicas de Parsing

**Conteúdo teórico**
- Análise de constituência vs. análise de dependências
- Chunking: identificação de NP, VP e PP
- Parsing de dependências: head, dependent e rótulos de relação
- Parsing de constituintes: gramáticas livres de contexto
- Algoritmos de parsing: CYK e Earley
- Roles semânticos e correferência: SRL e Winograd Schema

**Prática no Colab**
- Parsing de dependências com spaCy: `token.dep_`, `token.head`
- Visualização de árvore de dependências com `displacy` modo `dep`
- Chunking com NLTK `RegexpParser`
- Extração de triplas sujeito-verbo-objeto (SVO)
- Visualização gráfica de análise de constituência e dependências

**Ferramentas:** `spaCy` · `NLTK` · `displacy`

**Referências:** B1 Cap. 12–13 · C1 Cap. 7

</details>

---

### Módulo 4 — Redes Neurais e Transformers `semanas 10–11`

<details>
<summary><strong>Semana 10 · Redes Neurais para PLN — RNN, LSTM e GRU</strong></summary>

**Tópico do programa:** Tópico 13 — Arquiteturas para PLN (parte 1)

**Conteúdo teórico**
- Fundamentos de redes neurais: perceptron, camadas e funções de ativação
- Redes Recorrentes (RNN): processamento sequencial e memória
- O problema do gradiente que desaparece em sequências longas
- LSTM: células de memória e gates (input, forget, output)
- GRU: versão simplificada do LSTM com menos parâmetros
- BiLSTM: processamento bidirecional de sequências

**Prática no Colab**
- Ativação de GPU no Google Colab
- Keras: modelo `Sequential` com camadas `Embedding` + `LSTM`
- Tokenização e padding com `Keras Tokenizer` e `pad_sequences`
- Classificador de sentimento com LSTM
- Comparação LSTM vs. GRU em dataset de análise de sentimento
- Curvas de treino: loss e accuracy por época

**Ferramentas:** `TensorFlow/Keras` · `numpy` · `Colab GPU`

**Referências:** B2 Cap. 12–13 · B3 Cap. 3–4 · C2 Cap. 15

</details>

<details>
<summary><strong>Semana 11 · Transformers e Modelos Pré-treinados</strong></summary>

**Tópicos do programa:** Tópicos 10 e 13 — BERT, GPT, mecanismo de atenção

**Conteúdo teórico**
- Mecanismo de atenção: foco seletivo em sequências
- Arquitetura Transformer: encoder, decoder, self-attention e positional encoding
- BERT: pré-treinamento bidirecional (Masked LM) e fine-tuning
- GPT: geração autorregressiva e modelos causais
- Ecossistema Hugging Face: Hub, `pipeline`, tokenizadores e datasets
- Fine-tuning vs. few-shot vs. zero-shot learning

**Prática no Colab**
- API `pipeline` da Hugging Face: classificação e geração
- Classificação de sentimento com BERT em português (`neuralmind/bert-base-portuguese-cased`)
- Geração de texto com GPT-2
- Zero-shot classification com `facebook/bart-large-mnli`
- Comparação LSTM vs. BERT em tarefa de classificação

**Ferramentas:** `Hugging Face Transformers` · `PyTorch`

**Referências:** B3 Cap. 5–7 · C5 Cap. 1–3

</details>

---

### Módulo 5 — Aplicações e Ferramentas `semanas 12–14`

<details>
<summary><strong>Semana 12 · Aplicações: Tradução, Sumarização e Sentimento</strong></summary>

**Tópico do programa:** Tópico 11 — Aplicações de PLN

**Conteúdo teórico**
- Tradução automática neural (NMT): evolução de regras a Transformers
- Sumarização extrativa vs. abstrativa: diferenças e casos de uso
- Análise de sentimento: abordagem léxica (VADER) e supervisionada (BERT)
- ABSA — Aspect-Based Sentiment Analysis: sentimento por aspecto

**Prática no Colab**
- Tradução com Helsinki-NLP via Hugging Face (PT → EN e EN → PT)
- Sumarização abstrativa com `facebook/bart-large-cnn`
- Análise de sentimento com VADER e com BERT fine-tuned
- Pipeline integrado: coleta → pré-processamento → análise → visualização

**Ferramentas:** `Hugging Face` · `VADER` · `TextBlob`

**Referências:** B1 Cap. 14–15 · C5 Cap. 7 e 9 · C1 Cap. 8

</details>

<details>
<summary><strong>Semana 13 · Chatbots, Buscadores e Aplicações Multimodais</strong></summary>

**Tópicos do programa:** Tópicos 9 e 12 — Multimodalidade e Casos de Uso

**Conteúdo teórico**
- Chatbots baseados em regras vs. modelos de linguagem
- Gestão de diálogo: intenções, slots e contexto
- Motores de busca semântica: dense retrieval e busca vetorial
- Assistentes virtuais: pipeline ASR → NLU → NLG → TTS
- Recuperação de informação multimodal: texto, imagem e áudio
- Modelos cross-modal: CLIP e BLIP-2

**Prática no Colab**
- Chatbot de intenções com NLTK e classificação de texto
- Zero-shot classification com BART-MNLI para detecção de intenção
- Busca semântica com sentence-transformers e similaridade de cosseno
- Introdução a modelos CLIP para busca multimodal texto-imagem

**Ferramentas:** `NLTK` · `Hugging Face` · `sentence-transformers`

**Referências:** B1 Cap. 16 · C5 Cap. 11 e 14 · C1 Cap. 9

</details>

<details>
<summary><strong>Semana 14 · Ferramentas, Frameworks e Projeto Final</strong></summary>

**Tópico do programa:** Tópico 14 — Tecnologias e Ferramentas

**Conteúdo teórico**
- Panorama do ecossistema de PLN em 2026: spaCy, NLTK, Hugging Face, LangChain
- Frameworks de deep learning: TensorFlow/Keras e PyTorch
- Visão computacional integrada ao PLN: OpenCV e modelos multimodais
- Boas práticas: documentação de notebooks, reprodutibilidade e versionamento
- PLN em produção: APIs, latência, custo e ética
- Tendências: agentes autônomos, PLN em edge, modelos multilíngues

**Prática — Projeto Final**
- Apresentação dos projetos finais pelos alunos (5–10 min por grupo)
- Pipeline completo: coleta → pré-processamento → modelagem → avaliação → resultado
- Feedback coletivo e discussão sobre escolhas técnicas
- Compartilhamento dos notebooks no GitHub

**Ferramentas:** `spaCy` · `Hugging Face` · `TensorFlow` · `OpenCV`

**Referências:** B2 Cap. 14 · C4 Cap. 1 e 12 · C5 Cap. 15 · C1 Cap. 10

</details>

---

## Atividades avaliativas

| Atividade | Cobertura | Modalidade | Peso |
|-----------|-----------|------------|------|
| **AC1** — Atividade Complementar 1 | Semanas 1–7 | Laboratório individual (Colab) | 10% |
| **AP1** — Avaliação Parcial 1 | Semanas 1–7 | Prova escrita individual sem consulta | 35% |
| **AC2** — Atividade Complementar 2 | Semanas 8–13 | Laboratório individual (Colab) | 10% |
| **AP2** — Avaliação Parcial 2 | Semanas 8–13 | Prova escrita individual sem consulta | 35% |
| **Projeto Final** | Semanas 1–13 | Pipeline completo em grupo (apresentação) | 10% |

> As avaliações AP1 e AP2 são realizadas em papel, sem uso de computador ou consulta.  
> As atividades AC1 e AC2 são realizadas no laboratório com acesso ao Google Colab e à documentação das bibliotecas.

---

## Política de uso de Inteligência Artificial

A disciplina adota uma política progressiva de uso de IA ao longo das 14 semanas: nas semanas iniciais, o aluno observa demonstrações conduzidas pelo professor e usa IA apenas para formular dúvidas conceituais, sempre confrontando as respostas com a bibliografia do curso; nas semanas intermediárias, pode recorrer à IA como apoio à escrita de código, desde que compreenda cada linha produzida e seja capaz de justificá-la tecnicamente; nas semanas finais, a IA pode ser usada livremente como ferramenta de produção, desde que o aluno documente seu uso no relatório e demonstre domínio pleno do trabalho durante a apresentação. Em qualquer fase, são vedados: copiar respostas sem compreensão, citar referências sugeridas por IA sem verificar a fonte original, e apresentar conclusões que sejam mera paráfrase do que a ferramenta gerou. O critério definidor é simples — se o aluno não consegue explicar o que entregou, a IA substituiu o aprendizado em vez de apoiá-lo.

---

## Como usar este repositório

### 1. Abrir um notebook no Google Colab

Clique no ícone abaixo para abrir qualquer notebook diretamente no Colab sem instalar nada:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Ou navegue até a pasta da semana desejada, abra o arquivo `.ipynb` e clique em **"Open in Colab"** no topo da página do GitHub.

### 2. Clonar o repositório localmente

```bash
git clone https://github.com/seu-usuario/IBM3130-PLN-2026.git
cd IBM3130-PLN-2026
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` inclui todas as bibliotecas utilizadas ao longo do curso:

```
nltk>=3.8
spacy>=3.7
gensim>=4.3
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
tensorflow>=2.15
torch>=2.1
transformers>=4.38
sentence-transformers>=2.6
networkx>=3.2
```

### 4. Baixar o modelo de português do spaCy

```bash
python -m spacy download pt_core_news_sm
python -m spacy download pt_core_news_lg
```

---

## Referências bibliográficas

### Básicas

| Código | Referência |
|--------|-----------|
| **B1** | CASELI, H. M.; NUNES, M. G. V. (org.). *Processamento de linguagem natural: conceitos, técnicas e aplicações em português*. São Carlos: BPLN, 2023. |
| **B2** | FACELI, K. et al. *Inteligência Artificial: uma abordagem de aprendizado de máquina*. Rio de Janeiro: LTC, 2025. |
| **B3** | SUMATHI, S.; JANANI, M. *Neural Networks for Natural Language Processing*. Hershey, PA: Engineering Science Reference, 2020. |

### Complementares

| Código | Referência |
|--------|-----------|
| **C1** | MARTINS, J. S. et al. *Processamento de Linguagem Natural*. Porto Alegre: Grupo A, 2024. |
| **C2** | HAYKIN, S. *Redes Neurais: Princípios e Prática*. Porto Alegre: Bookman, 2007. |
| **C3** | SICSÚ, A. L.; SAMARTINI, A.; BARTH, N. L. *Técnicas de Machine Learning*. São Paulo: Blucher, 2023. |
| **C4** | NETTO, A.; MACIEL, F. *Python para Data Science e Machine Learning Descomplicado*. Rio de Janeiro: Alta Books, 2021. |
| **C5** | ROTHMAN, D. *Transformers for Natural Language Processing*. Birmingham: Packt Publishing, 2022. |

### Recursos online gratuitos

- [NLTK Book](https://www.nltk.org/book/) — Bird, Klein & Loper (2009)
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Jurafsky & Martin (2024)
- [Hugging Face Course](https://huggingface.co/learn/nlp-course) — curso gratuito de PLN com Transformers
- [spaCy 101](https://spacy.io/usage/spacy-101) — documentação interativa do spaCy
- [Universal Dependencies](https://universaldependencies.org/) — tagset UD para mais de 100 idiomas

---

## Temas sugeridos para o Projeto Final

Os grupos podem escolher um dos temas abaixo ou propor tema próprio (sujeito à aprovação até a Semana 12):

| # | Área | Tema |
|---|------|------|
| 01 | Análise de Sentimento | Classificador de sentimento em avaliações de produtos brasileiros |
| 02 | Análise de Sentimento | Monitoramento de sentimento em redes sociais sobre um tema atual |
| 03 | Extração de Informação | Extrator de entidades nomeadas em notícias jornalísticas |
| 04 | Extração de Informação | Detector de fake news com PLN |
| 05 | Sumarização | Sumarizador automático de artigos científicos |
| 06 | Sumarização | Resumo automático de atas de reunião |
| 07 | Chatbot | Chatbot de FAQ para instituição de ensino |
| 08 | Chatbot | Assistente de atendimento com classificação de intenções |
| 09 | Classificação | Classificador temático de artigos de notícias |
| 10 | Geração | Gerador de títulos com modelos pré-treinados |

---

## Contato

**Prof. Anderson Vanin**  
Ibmec — Ciência de Dados e Inteligência Artificial  
Disciplina IBM3130 — Processamento de Linguagem Natural  
2º Semestre de 2026

---

<div align="center">

Feito com dedicação para os estudantes de PLN do Ibmec · 2026

</div>
