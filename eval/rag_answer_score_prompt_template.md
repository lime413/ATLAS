# RAG Answer Score Prompt Template

We use this prompt for one evaluation sample, replace all placeholders before sending it to the LLM.

```text
You judge one result from a RAG question-answering system.

Task:
The system answers Natural Questions using retrieved Wikipedia pages.
The answer model is gemma-3-4b-it-Q4_K_M, size 4B.
The model must answer using only the retrieved pages.

Your goal:
Score how correct the system answer is compared with the gold answers and the provided evidence.

Score scale:
1 - completely wrong
2 - mostly wrong
3 - half wrong, half correct
4 - nuances are missed, but mostly correct
5 - correct

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

Write only one digit from 1 to 5.
Do not write any explanation.
```
