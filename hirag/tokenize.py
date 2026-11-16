"""Tokenizers for BM25.

- "jieba":  exactly what the authors' code does (list(jieba.cut(text))): no lowercasing,
            spaces and punctuation are kept as tokens.
- "whitespace": text.split() (LangChain BM25Retriever default, used in the authors' ToolLLM script).
- "clean":  lowercase, split tool names on "_", keep letters/digits only,
            drop English stopwords, Porter stemming.
"""
from __future__ import annotations

import re
from typing import Callable, List

_STEMMER = None
_STOP = None


def _jieba(text: str) -> List[str]:
    import jieba  # imported lazily: only needed for the authors' setting
    import logging
    jieba.setLogLevel(logging.ERROR)
    return list(jieba.cut(text))


def _whitespace(text: str) -> List[str]:
    return text.split()


def _clean(text: str) -> List[str]:
    global _STEMMER, _STOP
    if _STEMMER is None:
        from nltk.stem import PorterStemmer
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _STEMMER, _STOP = PorterStemmer(), ENGLISH_STOP_WORDS
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return [_STEMMER.stem(w) for w in words if w not in _STOP]


TOKENIZERS = {"jieba": _jieba, "whitespace": _whitespace, "clean": _clean}


def get_tokenizer(name: str) -> Callable[[str], List[str]]:
    if name not in TOKENIZERS:
        raise ValueError(f"unknown tokenizer {name!r}; choose from {list(TOKENIZERS)}")
    return TOKENIZERS[name]
