"""BERT WordPiece tokenization for Grounding DINO captions.

Uses the checkpoint's own ``tokenizer.json`` through the standalone
``tokenizers`` library, so ids and character offsets match the HF processor
without importing ``transformers``.
"""

import json
from typing import Sequence, Union

import torch
from huggingface_hub.errors import EntryNotFoundError
from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer as _HFTokenizer

from GDINOtorch.utils.hf_hub_utils import resolve_hub_file

PAD_TOKEN_ID = 0


def build_caption(phrases: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join phrases into a Grounding DINO caption: ``"cat. remote control."``.

    Phrases are lowercased and stripped of a trailing period, since ``.`` is the
    phrase delimiter the model was trained with. Returns the caption and each
    phrase's ``(start, end)`` character span within it.
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for phrase in phrases:
        phrase = phrase.strip().lower().rstrip(".").strip()
        if not phrase:
            raise ValueError("Grounding DINO cannot be prompted with an empty phrase")
        spans.append((cursor, cursor + len(phrase)))
        parts.append(phrase)
        cursor += len(phrase) + len(". ")
    return ". ".join(parts) + ".", spans


def _load_tokenizer(model_id: str):
    """The repo's ``tokenizer.json`` if it has one, else WordPiece over ``vocab.txt``.

    The MM-Grounding-DINO conversions ship only ``vocab.txt`` +
    ``tokenizer_config.json``; that is the slow ``BertTokenizer`` setup, which
    ``BertWordPieceTokenizer`` reproduces (normalization, [CLS]/[SEP]).
    """
    try:
        return _HFTokenizer.from_file(resolve_hub_file(model_id, "tokenizer.json"))
    except EntryNotFoundError:
        pass
    with open(resolve_hub_file(model_id, "tokenizer_config.json")) as f:
        config = json.load(f)
    return BertWordPieceTokenizer(
        resolve_hub_file(model_id, "vocab.txt"),
        lowercase=config.get("do_lower_case", True),
        strip_accents=config.get("strip_accents"),
        handle_chinese_chars=config.get("tokenize_chinese_chars", True),
    )


class Tokenizer:
    def __init__(self, model_id: str, max_length: int = 256):
        self.max_length = max_length
        self._tokenizer = _load_tokenizer(model_id)
        # The post-processor still adds [CLS]/[SEP] after truncation, matching
        # the HF processor's ``truncation=True``.
        self._tokenizer.enable_truncation(max_length)
        self._tokenizer.no_padding()

    def __call__(self, captions: Union[str, Sequence[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize one caption or a batch, padded to the longest.

        Returns ``input_ids`` and ``attention_mask`` (1 for real tokens), both
        ``[batch, seq_len]`` int64.
        """
        if isinstance(captions, str):
            captions = [captions]
        encodings = self._tokenizer.encode_batch(list(captions))
        seq_len = max(len(e.ids) for e in encodings)
        input_ids = torch.full((len(encodings), seq_len), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros((len(encodings), seq_len), dtype=torch.long)
        for i, e in enumerate(encodings):
            input_ids[i, : len(e.ids)] = torch.tensor(e.ids, dtype=torch.long)
            attention_mask[i, : len(e.ids)] = 1
        return input_ids, attention_mask

    def token_offsets(self, caption: str) -> list[tuple[int, int]]:
        """Character ``(start, end)`` of each token; ``(0, 0)`` for [CLS]/[SEP]."""
        return self._tokenizer.encode(caption).offsets

    def phrase_token_indices(
        self, caption: str, spans: Sequence[tuple[int, int]]
    ) -> list[list[int]]:
        """Token positions that fall inside each character span of ``caption``.

        Use with the spans from :func:`build_caption` to pool per-token logits
        back to per-phrase scores.
        """
        offsets = self.token_offsets(caption)
        indices: list[list[int]] = []
        for start, end in spans:
            tokens = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
            if not tokens:
                raise ValueError(
                    f"Phrase at caption[{start}:{end}] ({caption[start:end]!r}) matched no "
                    f"tokens; it may have been truncated at {self.max_length} tokens"
                )
            indices.append(tokens)
        return indices
