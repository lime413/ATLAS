# ATLAS - Project Workflow

ATLAS studies whether a RAG system can answer Natural Questions better when Wikipedia pages are represented as structured, LLM-friendly text instead of raw wiki text.

The local answer model is Gemma 3 4B GGUF served by `llama-server`.

## Setup

Install dependencies:

```bash
uv sync
```

Start the local model server:

```bash
llama-server \
  -m models/gemma3/gemma-3-4b-it-Q4_K_M.gguf \
  -ngl 99 \
  -c 8192 \
  --host 127.0.0.1 \
  --port 8082
```

The API URL is:

```text
http://127.0.0.1:8082/v1
```

## 1. Reduce the Dataset

The project starts from Natural Questions over Wikipedia.

First, filter questions that the LLM can answer without retrieval. These questions are not useful for testing RAG, so we keep only hard questions.

Script:

```text
dataset/filter_dataset.py
```

Example:

```bash
python dataset/filter_dataset.py \
  --input data/train.jsonl \
  --results data/filter_results.jsonl \
  --hard-records data/train_hard.jsonl \
  --base-url http://127.0.0.1:8082/v1 \
  --model gemma-3-4b-it-Q4_K_M.gguf \
  --use-bertscore
```

Then reduce the Wikipedia page database. The reduced DB keeps all pages referenced by the training questions and adds random pages until the target size is reached.

Script:

```text
dataset/reduce.py
```

Example:

```bash
python dataset/reduce.py \
  --train data/train.jsonl \
  --source-db data/wikipedia_pages.sqlite \
  --output-db data/wikipedia_pages_50k.sqlite \
  --target-pages 50000 \
  --seed 42
```

Main outputs:

```text
data/train_hard.jsonl
data/wikipedia_pages_50k.sqlite
```

## 2. Run LLM-Only Evaluation on Hard Questions

This evaluates Gemma without retrieval. It is the no-RAG baseline on hard questions.

Script:

```text
eval/run_llm_only_eval.py
```

Example:

```bash
python -u eval/run_llm_only_eval.py \
  --stage all \
  --input data/train_hard.jsonl \
  --output-dir output \
  --run-name llm_only_train_hard \
  --base-url http://127.0.0.1:8082/v1 \
  --model gemma-3-4b-it-Q4_K_M.gguf \
  --max-tokens 96 \
  --bert-device cuda:0
```

Outputs:

```text
output/llm_only_train_hard_answers.jsonl
output/llm_only_train_hard_results.jsonl
output/llm_only_train_hard_summary.json
```

## 3. Build the Raw-ish RAG Index

The first RAG baseline uses simple preprocessing of raw Wikipedia pages.

Script:

```text
rag/index.py
```

Raw-ish preprocessing:

- removes wiki bold/italic quotes;
- turns headings like `== History ==` into plain text;
- collapses repeated spaces and blank lines;
- keeps most page text close to raw Wikipedia format.

Build the index:

```bash
python rag/index.py \
  --db data/wikipedia_pages_50k.sqlite \
  --index-dir data/index_50k_rawish \
  --clean-mode raw-ish
```

The index uses two retrieval sources:

- dense retrieval with Jina v3 embeddings and FAISS;
- sparse retrieval with SQLite FTS5 and BM25 scoring.

At retrieval time, `rag/search.py` merges dense and sparse results with Reciprocal Rank Fusion. The default settings used in evaluation are:

```text
top_k = 5
dense_limit = 20
sparse_limit = 20
sparse_weight = 0.3
```

This means dense retrieval has weight `0.7`, and sparse retrieval has weight `0.3`.

Index files:

```text
data/index_50k_rawish/faiss.index
data/index_50k_rawish/passages.sqlite
data/index_50k_rawish/config.json
```

## 4. Evaluate RAG on Raw Pages

Script:

```text
eval/run_rag_hard_eval.py
```

Generate answers:

```bash
python -u eval/run_rag_hard_eval.py \
  --stage generate \
  --input data/train_hard.jsonl \
  --index-dir data/index_50k_rawish \
  --output-dir output \
  --run-name index_50k_rawish_50k_all \
  --base-url http://127.0.0.1:8082/v1 \
  --model gemma-3-4b-it-Q4_K_M.gguf \
  --top-k 5 \
  --dense-limit 20 \
  --sparse-limit 20 \
  --sparse-weight 0.3 \
  --max-tokens 96
```

Score answers:

```bash
python -u eval/run_rag_hard_eval.py \
  --stage score \
  --output-dir output \
  --run-name index_50k_rawish_50k_all \
  --bert-device cuda:0
```

Metrics:

- Exact Match;
- token F1;
- BERTScore F1.

BERTScore uses:

```text
microsoft/deberta-xlarge-mnli
```

Outputs:

```text
output/index_50k_rawish_50k_all_answers.jsonl
output/index_50k_rawish_50k_all_results.jsonl
output/index_50k_rawish_50k_all_summary.json
```

## 5. Analyse Raw RAG Mistakes

The mistake analysis asks the local LLM to explain the main cause of each error.

Script:

```text
eval/run_mistake_analysis.py
```

Prompt template:

```text
eval/rag_mistake_analysis_prompt_template.md
```

Example:

```bash
python eval/run_mistake_analysis.py \
  --input output/index_50k_rawish_50k_all_results.jsonl \
  --output output/index_50k_rawish_50k_all_mistake_analysis.json \
  --template eval/rag_mistake_analysis_prompt_template.md \
  --passages-db data/index_50k_rawish/passages.sqlite \
  --base-url http://127.0.0.1:8082/v1 \
  --model gemma-3-4b-it-Q4_K_M.gguf
```

The output field contains a short cause, for example:

```text
The retrieved chunks contain the answer, but the answer model missed it.
The gold reference page was not retrieved.
The question is ambiguous or not precise enough.
```

## 6. Score Raw RAG with LLM-as-Judge

The LLM-as-judge task gives each answer a score from 1 to 5.

Script:

```text
eval/run_mistake_analysis.py
```

Prompt template:

```text
eval/rag_answer_score_prompt_template.md
```

Score scale:

```text
1 - completely wrong
2 - mostly wrong
3 - half wrong, half correct
4 - nuances are missed, but mostly correct
5 - correct
```

Example:

```bash
python eval/run_mistake_analysis.py \
  --input output/index_50k_rawish_50k_all_results.jsonl \
  --output output/index_50k_rawish_50k_all_answer_scores.json \
  --template eval/rag_answer_score_prompt_template.md \
  --passages-db data/index_50k_rawish/passages.sqlite \
  --base-url http://127.0.0.1:8082/v1 \
  --model gemma-3-4b-it-Q4_K_M.gguf \
  --include-correct \
  --max-output-tokens 8
```

The output field contains only the judge score.

## 7. Build Structured Pages and Structured Index

The mistake analysis showed that many answers were present in retrieved chunks, but the model missed them. This suggested that raw chunks were too noisy or too hard to read.

So the next step is to convert raw Wikipedia pages into structured pages.

Script:

```text
dataset/structure_pages.py
```

Processing steps:

- remove comments, references, HTML tags, category links, file/image links, and noisy wiki syntax;
- unwrap wiki links, for example `[[A|B]]` becomes `B`;
- remove low-value sections such as `References`, `External links`, `See also`, `Notes`, and `Bibliography`;
- extract lead text and normal article sections;
- parse infobox fields into key-value pairs;
- parse wiki tables into captions, headers, and rows;
- convert infobox fields into readable fact lines;
- convert table rows into readable fact lines;
- build `clean_text` from title, facts, lead, and sections.

Generate structured pages:

```bash
python dataset/structure_pages.py \
  --db data/wikipedia_pages_50k.sqlite \
  --output data/structured_pages_50k.jsonl \
  --limit 0 \
  --max-table-rows 20 \
  --log-every 1000
```

Important fields:

```text
wikipedia_id
title
clean_text
facts
infobox_facts
table_facts
lead
sections
infobox
tables
stats
```

Build structured index:

```text
rag/index_structured.py
```

```bash
python rag/index_structured.py \
  --input data/structured_pages_50k.jsonl \
  --index-dir data/index_50k_structured
```

Compared with `rag/index.py`, the structured index adds:

- section and sentence chunks instead of fixed raw windows;
- page aliases and title disambiguation in every chunk;
- first lead sentence as page context in every chunk;
- compact page table of contents in every chunk;
- short fact chunks from infobox and table facts.

The structured index still writes:

```text
faiss.index
passages.sqlite
config.json
```

So the same retrieval code can load it with `--index-dir data/index_50k_structured`.

## 8. Evaluate RAG on Structured Pages

Use the same RAG evaluation script, but point it to the structured index.

Generate answers:

```bash
python -u eval/run_rag_hard_eval.py \
  --stage generate \
  --input data/train_hard.jsonl \
  --index-dir data/index_50k_structured \
  --output-dir output \
  --run-name index_50k_structured_50k_all \
  --base-url http://127.0.0.1:8082/v1 \
  --model gemma-3-4b-it-Q4_K_M.gguf \
  --top-k 5 \
  --dense-limit 20 \
  --sparse-limit 20 \
  --sparse-weight 0.3 \
  --max-tokens 96
```

Score answers:

```bash
python -u eval/run_rag_hard_eval.py \
  --stage score \
  --output-dir output \
  --run-name index_50k_structured_50k_all \
  --bert-device cuda:0
```

Then compare:

```text
output/index_50k_rawish_50k_all_summary.json
output/index_50k_structured_50k_all_summary.json
```

## Useful Files

```text
dataset/filter_dataset.py                      filter hard questions
dataset/reduce.py                              reduce Wikipedia DB to 50k pages
dataset/structure_pages.py                     build structured page JSONL
rag/index.py                                   build raw-ish FAISS + SQLite index
rag/index_structured.py                        build structured FAISS + SQLite index
rag/search.py                                  dense + sparse retrieval and RRF merge
eval/run_llm_only_eval.py                      LLM-only baseline
eval/run_rag_hard_eval.py                      RAG generation and scoring
eval/run_mistake_analysis.py                   mistake analysis and LLM-as-judge scoring
eval/rag_mistake_analysis_prompt_template.md   mistake cause prompt
eval/rag_answer_score_prompt_template.md       1-5 judge prompt
```
