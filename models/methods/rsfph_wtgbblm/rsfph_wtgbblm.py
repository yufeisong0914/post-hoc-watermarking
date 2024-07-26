from typing import Dict, Any
import torch
import hashlib

from scipy.stats import norm

import spacy
from nltk.tokenize import sent_tokenize

from .keyword_extractor import KeywordExtractor

from .mask_selector import MaskSelector

from ciwater_extrect_parals import lexical_substitute_transformer_model

import models.util_string as util_string
from models.wm_model import WatermarkModelForExistingText

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fairseq.models.transformer import TransformerModel


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


class WatermarkModel(WatermarkModelForExistingText):
    def __init__(
            self,
            watermark_message_type: str = 'zero-bit',
            use_z_test: bool = True, z_test_alpha: float = 0.05
    ):

        super().__init__(watermark_message_type, use_z_test, z_test_alpha)

        self.alpha = z_test_alpha
        self.z_alpha = norm.ppf(1 - self.alpha, loc=0, scale=1)
        self.p = 0.5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerModel.from_pretrained(
            'checkpoints/para/transformer/',
            'checkpoint_best.pt',
            bpe='subword_nmt',
            bpe_codes='checkpoints/para/transformer/codes.40000.bpe.en'
        ).cuda().eval()
        self.tokenizer = self.model

        self.nlp = spacy.load('en_core_web_sm')

        self.keyword_extractor = KeywordExtractor()
        self.mask_selector = MaskSelector(
            custom_keywords=["haojifei", "ciwater"], method="grammar", mask_order_by="dep", exclude_cc=False
        )
        self.keyword_extract_rate = 0.5
        self.replacement_rate = 1.0

    def watermark_text_generator(self, ori_paragraph) -> Dict[str, Any]:
        sentences = paragraph_tokenize(ori_paragraph)

        # q = 0  # todo:delete

        watermarked_text_all = ""
        all_wm_index = []
        for text in sentences:
            watermarked_text, watermark_index = self._watermark_embedding(text)
            watermarked_text_all = watermarked_text_all + watermarked_text + " "
            all_wm_index.append(watermark_index)  # dug的时候看看
            # q = q + len(watermark_embed_index)  # todo:delete

        # print(q)  # todo:delete

        watermarked_text_all = util_string.preprocess_string(watermarked_text_all)

        generator_result = {
            "watermarked_text": watermarked_text_all,
            "watermark_index": all_wm_index,
        }
        return generator_result

    def _watermark_embedding(self, text):
        doc_sentence = self.nlp(text)
        # 提取关键词
        all_entity_keywords, all_yake_keywords = self.keyword_extractor.extract_keyword(
            doc_sentence, self.keyword_extract_rate
        )
        # 根基关键词选择掩码位置(返回的mask_words是"spacy.tokens.token.Token"类型的list)
        mask_words_index, mask_words = self.mask_selector.return_mask(
            doc_sentence, all_entity_keywords=all_entity_keywords, all_yake_keywords=all_yake_keywords
        )
        # print(mask_words_index)
        if len(mask_words_index) == 0:
            return text
        else:
            len_keyword = len(all_entity_keywords) + len(all_yake_keywords)
            if self.replacement_rate < 5:
                mask_words_index = mask_words_index[:int(self.replacement_rate * len_keyword)]

        # 筛选真正需要替换的单词
        mask_words_index_new = []
        j = 0
        for i in mask_words_index:
            if binary_encoding_function(doc_sentence[i - 1].text + mask_words[j].text) == 0:
                mask_words_index_new.append(i)
            j = j + 1

        # return text, mask_words_index_new  # todo:delete

        # 为每一个maskword生成候选词
        all_candidates = []
        for i in mask_words_index_new:
            _, candidates = self._candidates_generator(doc_sentence, i)
            good_candidates_index = list(range(0, len(candidates)))  # 首先假设所有的词都是好词
            good_candidates_index = token_bit_filter(good_candidates_index, candidates, doc_sentence[i - 1].text)
            good_candidates_index = self._token_detectable_filter(good_candidates_index, candidates, doc_sentence, i)
            if good_candidates_index:
                # todo: 四个sim
                all_candidates.append(candidates[good_candidates_index[0]])
            else:
                all_candidates.append(doc_sentence[i].text)  # 如果经过两次过滤后没有候选词留下，那么将保留原词

        # 将候选词替换到原句中
        i = 0
        watermarked_list = [token.text for token in doc_sentence]
        for candidate in all_candidates:
            watermarked_list[mask_words_index_new[i]] = candidate
            i = i + 1

        # 将字符串数组转化为字符串
        watermarked_text = " ".join(watermarked_list).strip()

        return watermarked_text, mask_words_index_new

    def _candidates_generator(self, doc_sentence, mask_word_index):
        doc_tokens = [token.text for token in doc_sentence]

        # 前缀
        prefix = doc_tokens[0:mask_word_index]
        prefix_s = " ".join(prefix).strip()  # 合并为字符串
        prefix_s = util_string.preprocess_string(prefix_s)  # 处理句子前缀

        # mask
        mask_word = doc_tokens[mask_word_index]
        mask_word_info = [(
            token.text, token.lemma_, token.pos_, token.tag_,
            token.dep_, token.shape_, token.is_alpha, token.is_stop
        ) for j, token in enumerate(doc_sentence) if j == mask_word_index]

        # 后缀
        suffix = doc_tokens[mask_word_index + 1:]
        suffix_s = ""
        if len(suffix) >= 2:  # 选2个token作为后缀
            suffix_s = " ".join(suffix[:2]).strip()  # 合并为字符串
        else:
            suffix_s = " ".join(suffix[0:]).strip()
        suffix_s = util_string.preprocess_string(suffix_s)  # 处理句子后缀

        # 目标单词的词性
        # target_pos = main_word.split('.')[-1]
        # target_pos = pos_tag(ori_sen_words, tagset='universal')[i][1]
        # pos_coco = {'ADJ': 'a', 'ADJ-SAT': 's', 'ADV': 'r', 'NOUN': 'n', 'VERB': 'v'}
        # target_pos = pos_coco[target_pos]

        # 去除句子后面的引号或单引号
        # if suffix_s.endswith("\"") or suffix_s.endswith("'"):
        #     suffix_s = suffix_s[:-1]
        #     suffix_s = suffix_s.strip()

        # 目标单词的词形还原
        # target_lemma = lemmatize_word(target_word, target_pos=target_pos).lower().strip()
        # nltk.wordnet.WordNetLemmatizer()方法是NLTK库中的一个词形还原器类，用于将单词转换为它们的基本词形。
        # wordnet_lemmatizer = WordNetLemmatizer()
        # target_lemma = wordnet_lemmatizer.lemmatize(target_word, pos=target_pos).lower().strip()

        # bert_substitutes, bert_rank_substitutes, real_prev_scores, real_embed_scores
        outputs, candidates = (
            lexical_substitute_transformer_model(
                self.model,  # 模型
                doc_sentence.text,  # 原句
                prefix_s,  # 句子前缀
                mask_word,  # 目标单词
                # mask_word_info[0][2],  # 目标单词词性 Part-of-Speech
                # mask_word_info[0][3],  # 目标单词 词形还原（Lemmatization）
                suffix_s,
                100,
                -100
            )
        )
        return outputs, candidates

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
        temp_sentence_tokens = [token.text for token in doc_sentence]
        for index in good_candidates_index:
            temp_sentence_tokens[i] = candidates[index].strip("'")
            temp_sentence = " ".join(temp_sentence_tokens[:]).strip()
            temp_sentence = util_string.preprocess_string(temp_sentence)
            temp_doc_sentence = self.nlp(temp_sentence)
            # print(temp_sentence_doc.text)

            all_entity_keywords, all_yake_keywords = self.keyword_extractor.extract_keyword(
                temp_doc_sentence, self.keyword_extract_rate
            )
            mask_words_index, _ = self.mask_selector.return_mask(
                temp_doc_sentence, all_entity_keywords=all_entity_keywords, all_yake_keywords=all_yake_keywords
            )

            len_keyword = len(all_entity_keywords) + len(all_yake_keywords)
            mask_words_index = mask_words_index[:int(self.replacement_rate * len_keyword)]

            if i in mask_words_index:
                good_candidates_index_return.append(index)

        return good_candidates_index_return

    def _get_encodings(self, text):
        sentences = paragraph_tokenize(text)
        encodings, mask_words_index_list, mask_words_list = [], [], []
        for text in sentences:
            doc_text = self.nlp(text)
            # print(len(doc_text))
            # 提取关键词
            all_entity_keywords, all_yake_keywords = self.keyword_extractor.extract_keyword(
                doc_text, self.keyword_extract_rate
            )
            # 根基关键词选择掩码位置(返回的mask_words是"spacy.tokens.token.Token"类型的list)
            mask_words_index, mask_words = self.mask_selector.return_mask(
                doc_text, all_entity_keywords=all_entity_keywords, all_yake_keywords=all_yake_keywords
            )

            if self.replacement_rate < 5:
                len_keyword = len(all_entity_keywords) + len(all_yake_keywords)
                num_mask_words = int(self.replacement_rate * len_keyword)
                mask_words_index = mask_words_index[:num_mask_words]
                mask_words = mask_words[:num_mask_words]

            mask_words_index_list.append(mask_words_index)
            mask_words_list.append(mask_words)

            # print(mask_words_index)
            j = 0
            for k in mask_words_index:
                encodings.append(binary_encoding_function(doc_text[k - 1].text + mask_words[j].text))
                j = j + 1

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
    print('RS-FP-H_WTGBBLM')
    # wm_model = WatermarkModel()
    pass
