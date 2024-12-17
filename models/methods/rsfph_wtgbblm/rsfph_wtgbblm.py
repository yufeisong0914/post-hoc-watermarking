from typing import Dict, Any
import hashlib
import json
import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scipy.stats import norm
import torch
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token
from nltk.tokenize import sent_tokenize
import yake

from fairseq.models.transformer import TransformerModel
import models.util_string as util_string
from models.wm_model import WatermarkModelForExistingText
from models.methods.rsfph_wtgbblm.keyword_extractor import KeywordExtractor
from models.methods.rsfph_wtgbblm.mask_selector import MaskSelector
from models.utils.paraphraser import Paraphraser
from ciwater_extrect_parals import lexical_substitute_transformer_model


def binary_encoding_function(token):
    hash_value = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16)
    random_bit = hash_value % 2
    return random_bit


def token_bit_filter(good_candidates_index, candidates, pre_token):
    """
    Args:
        good_candidates_index: 好候选词的下标
        candidates: 候选词列表
        pre_token: 在原句中候选词的前一个词
    Returns: 经过筛选的好候选词下标
    """
    good_candidates_index_return = []
    for index in good_candidates_index:
        bit = binary_encoding_function(pre_token + candidates[index].lower())
        if bit == 1:
            good_candidates_index_return.append(index)

    return good_candidates_index_return


def paragraph_tokenize(paragraph):
    """
    将一段话分割为句子列表
    Args:
        paragraph: 一段话 "paragraph"
    Returns: 句子列表 ["sentence1", "sentence2", ...]
    """
    paragraph = util_string.preprocess_string(paragraph)  # 预处理一下
    # doc_paragraph = self.nlp(paragraph)
    # sentences = [sentence.text for sentence in doc_paragraph.sents]
    # return [s for s in sentences if s.strip()]
    return sent_tokenize(paragraph)


class RspfhWtgbblModel(WatermarkModelForExistingText):
    def __init__(
            self,  # paraphraser_model_name: str, paraphraser_model_root: str,
            paraphraser: Paraphraser,
            language: str = 'en', watermark_message_type: str = 'zero-bit',
            use_z_test: bool = True, z_test_alpha: float = 0.05
    ):

        super().__init__(language, watermark_message_type, use_z_test, z_test_alpha)

        self.paraphraser = paraphraser

        self.nlp = spacy.load('en_core_web_sm')

        self.entity_black_list = ["GPE", "ORG", "PERSON", "WORK_OF_ART", "EVENT"]
        self.yake_pos_black_list = ["NOUN", "PROPN"]
        self.yake_kwargs = {
            'lan': "en",
            "n": 1,
            "dedupLim": 0.9,
            "dedupFunc": 'seqm',
            "windowsSize": 1,
            "top": 20
        }

        # self.keyword_extractor = KeywordExtractor()
        self.mask_selector = MaskSelector(
            custom_keywords=["haojifei", "ciwater"], method="grammar", mask_order_by="dep", exclude_cc=False
        )

        self.keyword_extract_rate = 0.5
        self.replacement_rate = 1.0  # todo:9.9

    def _extract_entity_words(self, text_doc: Doc) -> list[int]:
        """
        提取文本（英文文本）中的所有命名实体（实体 type 需要在 entity_black_list 中）
        Args:
            text_doc:
        Returns:
            句子中的命名实体单词列表[在句子中的下标]
        """
        extracted = []
        for token in text_doc:
            if token.ent_type_ in self.entity_black_list:
                extracted.append(token.i)
        return extracted

    def _extract_yake_words(self, text_doc: Doc) -> list[int]:
        """
        使用 yake 方法提取文本（英文文本）中的关键词。
        Args:
            text_doc:
        Returns:
            句子中的yake关键词列表[在句子中的下标]
        """
        text = text_doc.text
        extractor = yake.KeywordExtractor(**self.yake_kwargs)
        r = extractor.extract_keywords(text)
        yake_words = [k[0] for k in r]

        extracted = []
        for token in text_doc:
            if token.text in yake_words:
                if token.pos_ in self.yake_pos_black_list:
                    extracted.append(token.i)
        return extracted

    def keywords_extractor(self, text_doc: Doc):
        """
        提取关键词
        Args:
            text_doc: 一段文本（英文文本）
        Returns:
            关键词
        """
        punctuation_removed = [token.text for token in text_doc if not token.is_punct]
        # 设置句子中keyword最多是多少
        keywords_max = int(self.keyword_extract_rate * len(punctuation_removed))

        entity_keywords_i = self._extract_entity_words(text_doc)
        yake_keywords_i = []
        if len(entity_keywords_i) > keywords_max:
            keywords = entity_keywords_i[:keywords_max]
        else:
            gap = keywords_max - len(entity_keywords_i)
            yake_keywords_i = self._extract_yake_words(text_doc)
            keywords = entity_keywords_i + yake_keywords_i[:gap]

        # return keywords
        return entity_keywords_i, yake_keywords_i

    def _watermark_embedding(self, text):  # todo: rename::get_watermark_position
        text_doc = self.nlp(text)

        # 提取关键词
        entity_keywords_index, yake_keywords_index = self.keywords_extractor(text_doc)
        entity_keywords = [text_doc[i].text for i in entity_keywords_index]
        yake_keywords = [text_doc[i].text for i in yake_keywords_index]

        # 根基关键词选择掩码位置(返回的mask_words是"spacy.tokens.token.Token"类型的list)
        mask_words_index = self.mask_selector.return_mask(
            text_doc, entity_keywords=entity_keywords, yake_keywords=yake_keywords
        )

        if len(mask_words_index) == 0:
            return text
        else:
            len_keyword = len(entity_keywords_index) + len(yake_keywords_index)
            if self.replacement_rate < 5:
                mask_words_index = mask_words_index[:int(self.replacement_rate * len_keyword)]

        # 筛选真正需要替换的单词
        mask_words_index_new = []
        for i in mask_words_index:
            if binary_encoding_function(text_doc[i - 1].text + text_doc[i].text) == 0:
                mask_words_index_new.append(i)

        text_words = [token.text for token in text_doc]
        # 为每一个maskword生成候选词
        all_candidates = []
        for i in mask_words_index_new:
            mask_word_candidates = self.paraphraser.substitutes_generator(text_words, i)
            good_candidates_index = list(range(0, len(mask_word_candidates)))  # 首先假设所有的词都是好词
            good_candidates_index = token_bit_filter(good_candidates_index, mask_word_candidates, text_doc[i - 1].text)
            good_candidates_index = self._token_detectable_filter(good_candidates_index, mask_word_candidates, text_doc, i)
            if good_candidates_index:
                # todo: 四个sim
                all_candidates.append(mask_word_candidates[good_candidates_index[0]])
            else:
                all_candidates.append(text_doc[i].text)  # 如果经过两次过滤后没有候选词留下，那么将保留原词

        # 将候选词替换到原句中
        i = 0
        watermarked_list = [token.text for token in text_doc]
        for candidate in all_candidates:
            watermarked_list[mask_words_index_new[i]] = candidate
            i = i + 1

        # 将字符串数组转化为字符串
        watermarked_text = " ".join(watermarked_list).strip()

        return watermarked_text, mask_words_index_new

    def watermark_text_generator(self, ori_paragraph) -> Dict[str, Any]:
        # sentences = paragraph_tokenize(ori_paragraph)
        ori_doc = self.nlp(ori_paragraph)

        watermarked_texts = []
        all_watermarks_index = []
        for text_doc in ori_doc.sents:
            watermarked_text, watermark_index = self._watermark_embedding(text_doc.text)
            watermarked_texts.append(watermarked_text)
            all_watermarks_index.append(watermark_index)  # dug的时候看看

        watermarked_text_all = " ".join(watermarked_texts)
        watermarked_text_all = util_string.preprocess_string(watermarked_text_all)

        generator_result = {
            "watermarked_text": watermarked_text_all,
            "watermark_index": all_watermarks_index,
        }
        return generator_result

    def _token_detectable_filter(self, good_candidates_index, candidates, doc_sentence, i):
        """
        Args:
            good_candidates_index: 好候选词的下标
            candidates: 候选词列表
            doc_sentence: 经过spaCy处理过的原句
            i: 候选词在原句中的位置
        Returns: 经过筛选的好候选词下标
        """
        good_candidates_index_return = []
        temp_text_tokens = [token.text for token in doc_sentence]
        for index in good_candidates_index:
            # 替换第i个单词
            temp_text_tokens[i] = candidates[index].strip("'")
            # 单词列表合并成字符串
            temp_text = " ".join(temp_text_tokens[:]).strip()
            temp_text = util_string.preprocess_string(temp_text)

            temp_text_doc = self.nlp(temp_text)

            entity_keywords_index, yake_keywords_index = self.keywords_extractor(temp_text_doc)
            entity_keywords = [temp_text_doc[i].text for i in entity_keywords_index]
            yake_keywords = [temp_text_doc[i].text for i in yake_keywords_index]

            mask_words_index = self.mask_selector.return_mask(
                temp_text_doc, entity_keywords=entity_keywords, yake_keywords=yake_keywords
            )

            len_keyword = len(entity_keywords) + len(yake_keywords)
            mask_words_index = mask_words_index[:int(self.replacement_rate * len_keyword)]

            if i in mask_words_index:
                good_candidates_index_return.append(index)

        return good_candidates_index_return

    def _get_encodings(self, text):
        sentences = paragraph_tokenize(text)
        encodings, mask_words_index_list, mask_words_list = [], [], []
        for text in sentences:
            text_doc = self.nlp(text)

            entity_keywords_index, yake_keywords_index = self.keywords_extractor(text_doc)
            entity_keywords = [text_doc[i].text for i in entity_keywords_index]
            yake_keywords = [text_doc[i].text for i in yake_keywords_index]

            # 根基关键词选择掩码位置(返回的mask_words是"spacy.tokens.token.Token"类型的list)
            mask_words_index = self.mask_selector.return_mask(
                text_doc, entity_keywords=entity_keywords, yake_keywords=yake_keywords
            )

            mask_words = [text_doc[i].text for i in mask_words_index]
            if self.replacement_rate < 5:
                len_keyword = len(entity_keywords) + len(yake_keywords)
                num_mask_words = int(self.replacement_rate * len_keyword)
                mask_words_index = mask_words_index[:num_mask_words]
                mask_words = mask_words[:num_mask_words]

            mask_words_index_list.append(mask_words_index)
            mask_words_list.append(mask_words)

            # print(mask_words_index)
            for i in mask_words_index:
                encodings.append(binary_encoding_function(text_doc[i - 1].text + text_doc[i].text))

        return encodings, mask_words_index_list, mask_words_list

    def watermark_text_detector(self, text) -> Dict[str, Any]:

        encodings, watermark_words_index, watermark_words = self._get_encodings(text)

        watermark_length = len(encodings)
        ones = sum(encodings)
        z = (ones - self.p * watermark_length) / (watermark_length * self.p * (1 - self.p)) ** 0.5

        p_value = norm.sf(z)
        # p_value = norm.sf(abs(z)) * 2
        is_watermarked = (z >= self.z_alpha)

        detector_result = {
            "watermarked": is_watermarked,
            "watermark_words": watermark_words,
            "watermark_words_index": watermark_words_index,
            "encoding": encodings,
            "ones/n": str(ones) + '/' + str(watermark_length),
            "z_score": z,
            "p_value": p_value,
        }

        # return is_watermark, p_value, watermark_length, ones, z, mask_words_index_list, mask_words_list
        return detector_result


if __name__ == "__main__":
    print('Robust and Semantic-Faithful Post-Hoc WTGBBLM')
    print('Author: Jifei Hao, et al.')
    print('Email: alfredwatson@foxmail.com')
