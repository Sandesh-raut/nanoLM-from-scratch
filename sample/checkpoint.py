"""
Phase 6 — Model checkpoint: save weights to disk, load them back.

Format: NumPy .npz (compressed) containing:
  - Architecture metadata as scalar arrays (__vocab_size, __embed_dim, ...)
  - Tokenizer vocab as a string array (__vocab)
  - All model weights keyed by sanitized param name (p__ prefix)

Why .npz?
---------
Pure NumPy, no dependency on PyTorch or pickle format.
Human-inspectable with np.load().  Portable across Python versions.
Compressed: a 100K-param model in float64 is ~0.8 MB uncompressed,
~0.4 MB compressed.

Usage
-----
  from sample.checkpoint import save, load

  # After training:
  save(model, tokenizer, 'runs/latest.npz')

  # To generate later:
  model, tokenizer = load('runs/latest.npz')
"""

from pathlib import Path
import numpy as np

from data.tokenizer import CharTokenizer


def _key(name: str) -> str:
    """Sanitize param name for use as an npz key (no dots, brackets)."""
    return 'p__' + name.replace('[', 'L').replace(']', 'R').replace('.', 'D')


def _arch_of(model) -> str:
    """'bigram' | 'transformer' | 'modern' — which class should rebuild this."""
    if hasattr(model, 'norm_type') and hasattr(model, 'pos_enc'):
        return 'modern'
    if getattr(model, 'n_layers', 0) == 0:
        return 'bigram'
    return 'transformer'


def save(model, tokenizer: CharTokenizer, path: str) -> str:
    """
    Save model weights + tokenizer vocab to a compressed .npz file.
    Returns the actual path written (adds .npz extension if missing).
    """
    path = str(path)
    if not path.endswith('.npz'):
        path += '.npz'
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    flat: dict = {}

    # Architecture metadata
    flat['__arch']       = np.array([_arch_of(model)],              dtype=object)
    flat['__vocab_size'] = np.array([model.vocab_size],             dtype=np.int32)
    flat['__embed_dim']  = np.array([model.embed_dim],              dtype=np.int32)
    flat['__block_size'] = np.array([getattr(model, 'block_size', 0)], dtype=np.int32)
    flat['__n_layers']   = np.array([getattr(model, 'n_layers',   0)], dtype=np.int32)
    flat['__n_heads']    = np.array([getattr(model, 'n_heads',    0)], dtype=np.int32)
    flat['__dropout']    = np.array([getattr(model, 'dropout',  0.0)], dtype=np.float64)

    # Phase 9 config — without these a modern model cannot be rebuilt, because
    # the norm / FFN / positional choice changes which parameters even exist.
    if _arch_of(model) == 'modern':
        flat['__n_kv_heads'] = np.array([model.n_kv_heads], dtype=np.int32)
        flat['__norm']       = np.array([model.norm_type],  dtype=object)
        flat['__ffn']        = np.array([model.ffn_type],   dtype=object)
        flat['__pos_enc']    = np.array([model.pos_enc],    dtype=object)
        flat['__moe']        = np.array([getattr(model, 'moe', None)], dtype=object)

        # Routing bias is updated by a rule rather than a gradient, so it never
        # appears in _flat_params — save it or a reloaded MoE model routes
        # differently from the one that was trained.
        for i, layer in enumerate(getattr(model, '_moe_layers', list)()):
            flat[f'__moe_bias_{i}'] = layer.expert_bias

    # Tokenizer vocab: sorted list of chars
    flat['__vocab'] = np.array(
        [tokenizer.itos[i] for i in range(tokenizer.vocab_size)],
        dtype=object,
    )

    # Model weights (in-order via _flat_params)
    for name, param in model._flat_params():
        flat[_key(name)] = param

    np.savez_compressed(path, **flat)
    return path


def load(path: str):
    """
    Load a checkpoint.  Returns (model, tokenizer).

    Rebuilds BigramModel, TransformerLM, or ModernTransformerLM, chosen by the
    '__arch' tag written at save time.  Checkpoints written before '__arch'
    existed fall back to the old n_layers rule (0 → bigram, else transformer).
    """
    data = np.load(path, allow_pickle=True)

    vocab_size = int(data['__vocab_size'][0])
    embed_dim  = int(data['__embed_dim'][0])
    block_size = int(data['__block_size'][0])
    n_layers   = int(data['__n_layers'][0])
    n_heads    = int(data['__n_heads'][0])
    dropout    = float(data['__dropout'][0])
    vocab      = [str(c) for c in data['__vocab']]

    # Reconstruct tokenizer (no text needed — vocab is stored)
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.vocab      = vocab
    tok.vocab_size = len(vocab)
    tok.stoi       = {c: i for i, c in enumerate(vocab)}
    tok.itos       = {i: c for i, c in enumerate(vocab)}

    # Reconstruct model
    arch = str(data['__arch'][0]) if '__arch' in data.files else (
        'bigram' if n_layers == 0 else 'transformer'
    )

    if arch == 'bigram':
        from model.bigram import BigramModel
        model = BigramModel(vocab_size=vocab_size, embed_dim=embed_dim)
    elif arch == 'modern':
        from model.modern_transformer import ModernTransformerLM
        model = ModernTransformerLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            block_size=block_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=int(data['__n_kv_heads'][0]),
            dropout=dropout,
            norm=str(data['__norm'][0]),
            ffn=str(data['__ffn'][0]),
            pos_enc=str(data['__pos_enc'][0]),
            moe=(data['__moe'][0] if '__moe' in data.files else None),
        )
        for i, layer in enumerate(model._moe_layers()):
            key = f'__moe_bias_{i}'
            if key in data.files:
                layer.expert_bias[:] = data[key]
    else:
        from model.transformer import TransformerLM
        model = TransformerLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            block_size=block_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

    # Assign weights in the same order _flat_params() yields them
    for name, param in model._flat_params():
        key = _key(name)
        if key not in data.files:
            raise KeyError(
                f"Checkpoint {path} is missing parameter '{name}'. It was most "
                f"likely written by a different architecture than '{arch}'."
            )
        saved = data[key]
        if saved.shape != param.shape:
            raise ValueError(
                f"Shape mismatch for '{name}' in {path}: checkpoint has "
                f"{saved.shape}, model expects {param.shape}."
            )
        param[:] = saved

    return model, tok
