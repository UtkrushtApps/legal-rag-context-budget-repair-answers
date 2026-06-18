# Solution Steps

1. Open `src/legal_rag/context_builder.py`; leave the retriever, Anthropic client wrapper, configuration, entry point, and tests unchanged.

2. Keep the existing `SYSTEM_PROMPT` intact so the model always receives the legal-research safety and citation instructions.

3. Add helper logic to sort retrieved chunks by descending relevance score, treating higher `score` values as more relevant.

4. Render retrieved evidence inside explicit XML markup: a top-level `<documents>` block containing one `<document index="..." score="...">` element per selected chunk, with nested `<citation>` and `<excerpt>` elements.

5. Render the user’s legal question outside the document block, for example in a separate `<question>...</question>` element, so the model can distinguish evidence from the user query.

6. Use the shared `count_tokens` function to count both the system prompt and the assembled user content; tests monkeypatch this function for deterministic assertions.

7. Start with all retrieved chunks selected. If the total token count exceeds `max_context_tokens`, repeatedly remove the final selected chunk, which is the least relevant remaining chunk after sorting.

8. Never truncate or modify the system prompt or user query. If the budget is impossibly small, return the minimal prompt with no evidence rather than damaging those required pieces.

9. Return the same payload shape as before: a dictionary with `system` and `user_content` keys.

10. Run `python -m pytest` to verify the prompt stays within budget when possible, trims low-scoring chunks first, uses XML document delimiters, keeps the question outside the evidence block, and continues to work with mocked Anthropic calls.

