from typing import Dict, Any
import openai
import numpy as np
import re
import spacy
from tqdm import tqdm

from models.wm_model import WatermarkModelForExistingText


def count_secret_words(text: str, words: list[str]) -> float:
    """
    Count the number of secret words in text.
    Args:
        text:
        words:

    Returns:

    """
    green_words_num = 0
    text_words = text.split()  # todo: spacy split?
    for word in words:
        if word in text_words:  # todo: 个数？
            green_words_num += 1
    return green_words_num / len(words)


def cal_cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
    # 计算两个向量的点积
    dot_product = np.dot(vector1, vector2)

    # 计算两个向量的模长
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)

    # 计算余弦相似度
    cos_similarity = dot_product / (norm_a * norm_b)

    return cos_similarity


def extract_floats_from_string(s):
    # 使用正则表达式匹配所有的浮点数
    float_pattern = r"[-+]?\d*\.\d+"
    floats = re.findall(float_pattern, s)

    # 将提取到的字符串转换成浮点数
    float_list = [float(num) for num in floats]

    return float_list


class PostmarkModel(WatermarkModelForExistingText):
    def __init__(
            self, embedder_model_name: str, inserter_model_name: str, secret_words_table_root: str,
            embedder_model_open_source: bool = False, embedder_model_root: str = None,
            embedder_model_key: str = None, embedder_model_base_url: str = None,
            similarity_threshold: float = 0.7,
            inserter_model_open_source: bool = False, inserter_model_root: str = None,
            inserter_model_key: str = None, inserter_model_base_url: str = None,
            insertion_ratio: int = 0.12,
            watermark_message_type: str = 'zero-bit',
    ):
        super().__init__(watermark_message_type)

        # "any embedding model can be used here"
        # TEXT-EMBEDDING-3-LARGE (OpenAI, 2024b)
        # NOMIC-EMBED (Nussbaum et al., 2024)
        self.embedder_model_name = embedder_model_name

        self.embedder_model_open_source = embedder_model_open_source
        if embedder_model_open_source:  # 如果模型开源
            if embedder_model_root:  # 如果给出模型本地路径
                self.embedder_model_root = embedder_model_root
            else:
                self.embedder_model_root = embedder_model_name
            self.embedder_model = None  # todo: load model
        else:  # 如果模型不开源
            self.embedder_model_key = embedder_model_key
            self.embedder_model_base_url = embedder_model_base_url

        # 加载秘密单词表
        self.secret_words_table_root = secret_words_table_root
        self.similarity_threshold = similarity_threshold

        # GPT-4O (OpenAI)
        # LLAMA-3-70B-INST (AI@Meta, 2024)
        self.inserter_model_name = inserter_model_name

        self.inserter_model_open_source = inserter_model_open_source
        if inserter_model_open_source:  # 如果模型开源
            if inserter_model_root:
                self.inserter_model_root = inserter_model_root
            else:
                self.inserter_model_root = inserter_model_name
            self.inserter_model = None  # todo: load model
        else:
            self.inserter_model_key = inserter_model_key
            self.inserter_model_base_url = inserter_model_base_url

        # The insertion ratio represents the percentage of the input text’s word count.
        self.insertion_ratio = insertion_ratio
        self.nlp = spacy.load('en_core_web_sm')

    def _embedding_from_api(self, text: str) -> list[float]:
        client = openai.OpenAI(
            api_key=self.embedder_model_key,
            base_url=self.embedder_model_base_url,
        )
        response = client.embeddings.create(
            model=self.embedder_model_name,
            input=text,
            encoding_format="float"
        )
        embedding_list = response.data[0].embedding
        return embedding_list

    def _embedding_from_open_source_model(self, text: str) -> list[float]:
        # todo
        pass

    def _embedder(self, text: str, embed_level: str = 'sentence') -> list[float]:
        """
        The EMBEDDER needs to be capable of projecting both words and documents into a high-dimensional latent space.
        Args:
            text: the input text(a sentence or a paragraph or a document)
        Returns:
            a high-dimensional vector of the input text.
        """
        if self.embedder_model_open_source:
            embedding_list = self._embedding_from_open_source_model(text)
        else:
            embedding_list = self._embedding_from_api(text)
        return embedding_list

    def _cal_insert_table(self, vector: list[float]) -> list[str]:
        """
        Returns:
            Insert word table (list).
        """
        insert_table = []
        file = open(self.secret_words_table_root, 'r')  # todo: init load

        lines = file.readlines()
        bar = tqdm(total=len(lines))
        for line in lines:
            secret_word_embedding = extract_floats_from_string(line)
            sim = cal_cosine_similarity(vector, secret_word_embedding)
            if sim > self.similarity_threshold:
                word = line.split(':')[0]
                insert_table.append((word, sim))
            # else:
            #     word = line.split(':')[0]
            #     print(word, sim)
            bar.update(1)
        file.close()

        # 对列表进行排序，按照元组中的浮点数排序
        insert_table_sorted = sorted(insert_table, key=lambda x: x[1], reverse=True)
        insert_table_word = [item[0] for item in insert_table_sorted]

        return insert_table_word

    def _get_insert_table_top(self, text: str, insert_table: list[str]) -> list[str]:
        doc_text = self.nlp(text)
        punctuation_removed = [token.text for token in doc_text if not token.is_punct]  # 去除标点符号
        top_n = len(punctuation_removed) * self.insertion_ratio
        return insert_table[:round(top_n)]

    def _inserting_from_api(self, text: str, words: list[str]) -> str:
        # word_list = ''
        # for word in words:
        #     word_list += word + ', '
        word_list = str(words)

        client = openai.OpenAI(
            api_key=self.inserter_model_key,
            base_url=self.inserter_model_base_url,
        )

        response = client.chat.completions.create(
            model=self.inserter_model_name,
            messages=[{
                "role": "system",
                "content": "Given below are a piece of text and a word list. "
                           "Rewrite the text to incorporate all words from the provided word list. "
                           "The rewritten text must be coherent and factual. "
                           "Distribute the words from the list evenly throughout the text, "
                           "rather than clustering them in a single section. "
                           "When rewriting the text, try your best to minimize text length increase. "
                           "Only return the rewritten text in your response, do not say anything else."
            }, {
                "role": "user",
                "content": "Text: {" + text + "} , Word list: " + word_list + ",  Rewritten text:"
            }
            ]
        )
        watermarked_text = response.choices[0].message.content
        return watermarked_text

    def _inserting_from_open_source_model(self, text: str, words: list[str]) -> str:
        # todo
        pass

    def _inserter(self, text: str, words: list[str]) -> str:
        """
        The INSERTER needs to have instruction-following capabilities, and its purpose is to rewrite the input text
        to incorporate words from the watermark word list.
        Args:
            text: the input text
            words: the watermark word list
        Returns:
            Watermarked Text.
        """
        if self.inserter_model_open_source:
            watermarked_text = self._inserting_from_open_source_model(text, words)
        else:
            watermarked_text = self._inserting_from_api(text, words)
        return watermarked_text

    def watermark_text_generator(self, text: str) -> Dict[str, Any]:
        vector = self._embedder(text)
        insert_table = self._cal_insert_table(vector)
        insert_table_top = self._get_insert_table_top(text, insert_table)
        watermarked_text = self._inserter(text, insert_table_top)
        generator_result = {
            "watermarked_text": watermarked_text,
            "watermarked_text_embedding": vector,
            "embedding_words": insert_table_top,
        }
        return generator_result

    def watermark_text_detector(self, text: str) -> Dict[str, Any]:
        vector = self._embedder(text)
        insert_table = self._cal_insert_table(vector)
        insert_table_top = self._get_insert_table_top(text, insert_table)
        p = count_secret_words(text, insert_table_top)

        if p > 0.5:
            watermarked = True
        else:
            watermarked = False

        detector_result = {
            "watermarked": watermarked,
            "watermarked_text_embedding": vector,
            "embedding_words": insert_table_top,
            "p_value": p,
        }
        return detector_result

# if __name__ == '__main__':
#     vec1 = [-0.015056877, -0.017690022, -0.018006723, 0.018766806, -0.0043817866]
#     vec2 = [0.015056877, 0.017690022, 0.018006723, -0.018766806, 0.0043817866]
#
#     # 计算余弦相似度
#     similarity = cal_cosine_similarity(vec1, vec2)
#     print(similarity)
