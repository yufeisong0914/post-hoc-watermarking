from abc import ABC, abstractmethod
from typing import Dict, Any
import json

import requests
from fairseq.models.transformer import TransformerModel


class Paraphraser:
    def __init__(self, lan: str = "en"):
        self.lan = lan

    @abstractmethod
    def substitutes_generator(self, text_words: list[str], mask_word_index: int, top_k: int = None) -> list[str]:
        pass


class LspgParaphraser(Paraphraser):
    def __init__(self, lan: str = "en"):
        super().__init__(lan)

    def substitutes_generator(self, text_words: list[str], mask_word_index: int, top_k: int = None) -> list[str]:
        prefix = text_words[:mask_word_index]
        prefix_s = " ".join(prefix).strip()

        target = text_words[mask_word_index]

        suffix = text_words[mask_word_index + 1:]
        suffix_s = " ".join(suffix).strip()

        response = requests.get(
            f"http://172.25.220.177:8085/get_candidates?prefix={prefix_s}&target={target}&suffix={suffix_s}"
        )

        candidates = []
        if response.status_code == 200:
            r = response.text
            j = json.loads(r.strip('\n'))
            candidates = j["substitutes"]
            if top_k is not None:
                candidates = candidates[:top_k]
        else:
            print(f"ciwater error: {response.status_code}")

        return candidates


class ParalsParaphraser(Paraphraser):
    def __init__(self, lan: str = "en"):
        super().__init__(lan)

    def substitutes_generator(self, text_words: list[str], mask_word_index: int, top_k: int = None) -> list[str]:
        prefix = text_words[:mask_word_index]
        prefix_s = " ".join(prefix).strip()

        target = text_words[mask_word_index]

        suffix = text_words[mask_word_index + 1:mask_word_index + 3]
        suffix_s = " ".join(suffix).strip()

        response = requests.get(
            f"http://172.25.220.177:8084/get_candidates?prefix={prefix_s}&target={target}&suffix={suffix_s}"
        )

        candidates = []
        if response.status_code == 200:
            r = response.text
            j = json.loads(r.strip('\n'))
            candidates = j["substitutes"]
            if top_k is not None:
                candidates = candidates[:top_k]
        else:
            print(f"ciwater error: {response.status_code}")

        return candidates
