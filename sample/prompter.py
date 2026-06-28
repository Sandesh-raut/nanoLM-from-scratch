"""
Phase 8 — Prompter: system prompt injection and chat format.

Wraps the instruction template so the rest of the codebase doesn't
have to know the exact template syntax.

Usage
-----
  from sample.prompter import Prompter

  p = Prompter("You are nanoLM, a tiny but helpful assistant.")
  prompt   = p.build("What is your name?")    # formatted input for generate()
  response = p.extract(generated_text)        # strip prefix, return response only

Maps to
-------
  Every LLM API (OpenAI, Anthropic, Gemini) has a system-prompt field.
  Under the hood they inject it using exactly this pattern: prepend a
  formatted block before the user's message, then let the model complete
  from the 'assistant:' marker onward.
"""

from data.instruct_dataset import (
    format_prompt,
    RESPONSE_START,
    RESPONSE_END,
    DEFAULT_SYSTEM,
)


class Prompter:
    """
    Thin wrapper that holds a system prompt and formats user messages.

    Parameters
    ----------
    system : str
        The system prompt. Injected before every user message.
        Defaults to DEFAULT_SYSTEM if not supplied.
    """

    def __init__(self, system: str = DEFAULT_SYSTEM):
        self.system = system

    def build(self, user: str) -> str:
        """
        Format a user message into a model-ready prompt string.

        The returned string ends at '### Assistant:\\n' so that
        generate() will continue from that point.

        Example
        -------
        >>> p = Prompter("Be concise.")
        >>> p.build("Say hello.")
        '### System:\\nBe concise.\\n### User:\\nSay hello.\\n### Assistant:\\n'
        """
        return format_prompt(system=self.system, user=user)

    def extract(self, generated: str) -> str:
        """
        Given the full generated text (prompt + response), return only
        the assistant's response — everything between '### Assistant:\\n'
        and '\\n### End' (or end of string).

        Parameters
        ----------
        generated : str
            Full output from generate(), including the seed prompt.

        Returns
        -------
        str
            Just the assistant's response, stripped of leading/trailing whitespace.

        Example
        -------
        >>> p = Prompter()
        >>> p.extract("### System:\\n...\\n### Assistant:\\nHello!\\n### End\\n")
        'Hello!'
        """
        # Find the last occurrence of RESPONSE_START (the model's completion)
        idx = generated.rfind(RESPONSE_START)
        if idx == -1:
            return generated.strip()
        response = generated[idx + len(RESPONSE_START):]
        # Strip everything from ### End onward
        end_idx = response.find(RESPONSE_END.lstrip('\n'))
        if end_idx != -1:
            response = response[:end_idx]
        return response.strip()

    def chat(self, model, tokenizer, user: str,
             max_new_tokens: int = 80,
             temperature: float = 0.8,
             top_k: int = 0,
             top_p: float = 1.0) -> str:
        """
        One-shot chat: build prompt → generate → extract response.

        Parameters
        ----------
        model        : trained nanoLM model (NumPy or PyTorch)
        tokenizer    : CharTokenizer from the SFT run
        user         : the user's message
        max_new_tokens, temperature, top_k, top_p : generation params

        Returns
        -------
        str : assistant response only
        """
        from sample.sampler import generate

        block_size = getattr(model, 'block_size', 64)
        prompt     = self.build(user)
        generated  = generate(
            model, tokenizer, seed_text=prompt,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return self.extract(generated)

    def __repr__(self) -> str:
        preview = self.system[:40] + ('...' if len(self.system) > 40 else '')
        return f"Prompter(system='{preview}')"
