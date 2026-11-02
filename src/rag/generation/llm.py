"""Provider-agnostic generation interface + a dependency-free echo model.

The default `EchoLLM` synthesizes a deterministic answer from the retrieved
context (no API key needed) so the pipeline is runnable and testable. Swap to
a real provider (Anthropic, OpenAI, Bedrock, Azure) under providers/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag.types import ScoredChunk


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


def load_prompt_template(cfg) -> str:
    path = Path(cfg._repo_root) / cfg.paths.prompts_dir / "answer.txt"
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, question: str, chunks: list[ScoredChunk]) -> str:
    context = "\n\n".join(f"[{c.chunk.source}] {c.chunk.text}" for c in chunks)
    return template.format(context=context or "(no context found)", question=question)


class EchoLLM:
    """Deterministic placeholder: extracts the most on-topic sentence from
    context and returns it with a citation. Stands in for a real LLM so the
    system works offline.
    """

    def __init__(self, model: str = "placeholder-1"):
        self.model = model

    def generate(self, prompt: str) -> str:
        # The prompt embeds Context then Question; pull them back out.
        question = ""
        if "# Question" in prompt:
            question = prompt.split("# Question", 1)[1].split("# Answer", 1)[0].strip()

        context_block = ""
        if "# Context" in prompt:
            context_block = prompt.split("# Context", 1)[1].split("# Question", 1)[0].strip()

        if not context_block or context_block == "(no context found)":
            return "I don't know based on the available context."

        q_terms = {w.lower().strip("?.,") for w in question.split()}
        best, best_overlap = "", -1
        for line in context_block.split("\n"):
            for sentence in line.split(". "):
                terms = {w.lower().strip("?.,") for w in sentence.split()}
                overlap = len(q_terms & terms)
                if overlap > best_overlap and len(sentence.split()) > 3:
                    best, best_overlap = sentence.strip(), overlap
        return best or context_block.split("\n")[0]


def get_llm(cfg) -> LLM:
    provider = cfg.generation.provider
    if provider == "echo":
        return EchoLLM(model=cfg.generation.model)
    # if provider == "anthropic":
    #     from rag.generation.providers.anthropic_llm import AnthropicLLM
    #     return AnthropicLLM(model=cfg.generation.model, ...)
    raise ValueError(f"Unknown generation provider: {provider!r}")
