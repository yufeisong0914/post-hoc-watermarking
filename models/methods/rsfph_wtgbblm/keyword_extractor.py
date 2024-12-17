import re

from datasets import load_from_disk
import yake
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from .mask_selector import MaskSelector


# Find Proper Noun on sentence level.
# Compute Yake on sentence level.
# Compute Tf-idf on sample level

# Sort them on sentence level.
# Identify any proper noun or entity per sentence
# If exists, then pick them as keywords first
# if space remains, put the other keywords in descending order


class KeywordExtractor:

    def __init__(self):
        self.yake_kwargs = {
            'lan': "en",
            "n": 1,
            "dedupLim": 0.9,
            "dedupFunc": 'seqm',
            "windowsSize": 1,
            "top": 20
        }
        self.entity_black_list = ["GPE", "ORG", "PERSON", "WORK_OF_ART", "EVENT"]
        self.yake_pos_black_list = ["NOUN", "PROPN"]

    def extract_keyword(self, doc_text: Doc, keyword_extract_rate: float):
        """
        提取句子中的关键词（“关键词"是句子中不太可能被修改的词）
        Args:
            doc_text: 经过spaCy模型处理过的文本对象doc
            keyword_extract_rate: 率
        Returns:
            命名实体关键词列表 和 yake关键词列表
        """
        # 去除标点符号
        punctuation_removed = [token.text for token in doc_text if not token.is_punct]
        # 设置句子中keyword最多是多少
        num_keyword_max = max(1, int(keyword_extract_rate * len(punctuation_removed)))
        # 句子中初始的keyword数为0
        keyword_per_sentence = 0

        # List[(index, token, IOB to determine redundant or not), ...]
        entity_keywords = self._extract_entity_word(doc_text)  # 提取命名实体
        i = 0
        all_entity_keywords = []
        while i < len(entity_keywords) and keyword_per_sentence < num_keyword_max:
            all_entity_keywords.append(entity_keywords[i][1])
            i += 1
            keyword_per_sentence += 1

        # List[(token, score), ...]
        yake_keywords = self._extract_yake_word(doc_text)
        i = 0
        all_yake_keywords = []
        while i < len(yake_keywords) and keyword_per_sentence < num_keyword_max:
            all_yake_keywords.append(yake_keywords[i][0])
            i += 1
            keyword_per_sentence += 1

        return all_entity_keywords, all_yake_keywords

    def _extract_entity_word(self, doc_sentence):
        """
        提取句子中的在 entity_black_list 中的单词
        Args:
            doc_sentence: 经过spaCy模型处理过的句子对象doc
        Returns: 句子中的命名实体单词列表
        """
        extracted = []
        for i, token in enumerate(doc_sentence):
            # print(token.ent_type_)
            if token.ent_type_ in self.entity_black_list:
                extracted.append((i, token, token.ent_type_))

        return extracted

    def _extract_yake_word(self, doc_text):
        """
        提取句子中的经过yake计算后的关键单词
        Args:
            doc_text: 经过spaCy模型处理过的句子对象doc
        Returns: 由yake方法计算出的关键词列表
        """
        extracted = []
        text = doc_text.text
        kw_extractor = yake.KeywordExtractor(**self.yake_kwargs)
        keyword = kw_extractor.extract_keywords(text)  # list[('token', score), ('token', score), ...]
        keyword_list = [k[0] for k in keyword]

        for token in doc_text:
            if token.text in keyword_list and token.pos_ in self.yake_pos_black_list:
                idx = keyword_list.index(token.text)
                extracted.append([token, keyword[idx][1]])
                keyword_list.remove(token.text)
        return extracted


def preprocess_txt(corpus):
    corpus = [t.replace("\n", " ") for t in corpus]
    corpus = [t.replace("\t", " ") for t in corpus]
    # html tag
    CLEANR = re.compile('<.*?>')
    corpus = [re.sub(CLEANR, '', c) for c in corpus if len(c) > 0]
    corpus = [re.sub(r"\.+", ".", c) for c in corpus]
    return corpus


if __name__ == "__main__":
    dtype = '../dataset/clean/ciwater_test_ours_data2'
    corpus = load_from_disk(dtype)['train']['text']
    corpus = preprocess_txt(corpus)

    keyword_extractor_ciwater = KeywordExtractor()
    mask_selector_ciwater = MaskSelector(custom_keywords=['101'], method='grammar', mask_order_by='dep')
    all_keywords, all_entity_keywords = [], []

    spacy_model = spacy.load("en_core_web_sm")

    sentences = []
    for paragraph in corpus:
        doc_paragraph = spacy_model(paragraph.strip())
        for sent in doc_paragraph.sents:
            sentences.append(sent.text)
        # print(sentences)
        for sentence in sentences:
            doc_sentence = spacy_model(sentence.strip())
            keywords, entity_keywords = keyword_extractor_ciwater.extract_keyword(doc_sentence, 0.5)
            all_keywords.append(keywords)
            all_entity_keywords.append(entity_keywords)
            mask_idx, mask_word = mask_selector_ciwater.return_mask(
                doc_sentence, entity_keywords=keywords, yake_keywords=entity_keywords
            )
            # print(doc_sentence)
        sentences = []

    print(all_keywords)
    print(all_entity_keywords)
