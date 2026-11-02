Guardrails protect a RAG system on both the input and the output side.

On input, the system screens the query for prompt injection and personally identifiable information before any retrieval happens. On output, a faithfulness or grounding gate checks that the generated answer is actually supported by the retrieved context. If the grounding score is too low, the answer is blocked and replaced with a safe fallback.

A guardrail implemented as a gate in the code path is stronger than an instruction written in the prompt, because a crafted input can argue against an instruction but cannot bypass a gate.
