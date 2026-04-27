# RAG Mistake Analysis Prompt Template

We use this prompt for one evaluation sample, replace all placeholders before sending it to the LLM.

```text
You analyse one result from a RAG question-answering system.

Task:
The system answers Natural Questions using retrieved Wikipedia pages.
The answer model is gemma-3-4b-it-Q4_K_M, size 4B.
The model must answer using only the retrieved pages.

Your goal:
Explain the most likely cause of the mistake, or say that the answer is correct.

Possible causes (output examples):
- The gold reference page was not retrieved.
- The gold reference page does not contain answer.
- The retrieved chunks do not contain enough information.
- The retrieved chunks contain the answer, but the answer model missed it.
- The question is ambiguous or not precise enough.
- The gold answer is incomplete or too strict.
- The model answer is correct, but the automatic metric likely marked it wrong.
- The answer model may be too small for this case.

Question:
{QUESTION}

Gold answers:
{GOLD_ANSWERS}

Gold reference chunks:
{GOLD_REFERENCE_PAGES}

Retrieved chunks:
{RETRIEVED_PAGES}

System answer:
{SYSTEM_ANSWER}

Evaluation metrics:
Exact match: {EXACT_MATCH}
Token F1: {TOKEN_F1}
BERTScore F1: {BERTSCORE_F1}

Write only one short sentence.
Do not write more than 25 words.
Be precise and name the main cause.
```
