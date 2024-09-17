from typing import Dict, Any
import openai
import numpy as np
import re
import spacy
from tqdm import tqdm

from models.wm_model import WatermarkModelForExistingText


def cal_cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
    # 计算两个向量的点积
    dot_product = np.dot(vector1, vector2)

    # 计算两个向量的模长
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)

    # 计算余弦相似度
    cos_similarity = dot_product / (norm_a * norm_b)

    return cos_similarity


def extract_floats_from_string(s) -> list[float]:
    # 使用正则表达式匹配所有的浮点数
    float_pattern = r"[-+]?\d*\.\d+"
    floats = re.findall(float_pattern, s)

    # 将提取到的字符串转换成浮点数
    float_list = [float(num) for num in floats]

    return float_list


class PostmarkModel(WatermarkModelForExistingText):
    def __init__(
            self, embedder_model_name: str, inserter_model_name: str,
            secret_words_table_path: str, load_secret_words_table: bool = True,
            embedder_model_open_source: bool = False, embedder_model_path: str = None,
            embedder_model_key: str = None, embedder_model_base_url: str = None,
            inserter_model_open_source: bool = False, inserter_model_path: str = None,
            inserter_model_key: str = None, inserter_model_base_url: str = None,
            insertion_ratio: int = 0.12, similarity_threshold: float = 0.7,
            language: str = 'en', watermark_message_type: str = 'zero-bit',
            use_z_test: bool = False,
    ):
        super().__init__(language, watermark_message_type, use_z_test)

        # "any embedding model can be used here"
        # TEXT-EMBEDDING-3-LARGE (OpenAI, 2024b)
        # NOMIC-EMBED (Nussbaum et al., 2024)
        self.embedder_model_name = embedder_model_name

        self.embedder_model_open_source = embedder_model_open_source
        if embedder_model_open_source:  # 如果模型开源
            if embedder_model_path:  # 如果给出模型本地路径
                self.embedder_model_root = embedder_model_path
            else:
                self.embedder_model_root = embedder_model_name
            self.embedder_model = None  # todo: load model
        else:  # 如果模型不开源
            self.embedder_model_key = embedder_model_key
            self.embedder_model_base_url = embedder_model_base_url

        # 插入表
        self.secret_words_table_path = secret_words_table_path
        if load_secret_words_table:
            self.secret_words_table: list[tuple[str, list[float]]] = self._load_secret_table()
        else:
            self.secret_words_table = None
        # todo: sql
        self.new_words_table_runtime: list[tuple[str, list[float]]] = []

        # GPT-4O (OpenAI)
        # LLAMA-3-70B-INST (AI@Meta, 2024)
        self.inserter_model_name = inserter_model_name

        self.inserter_model_open_source = inserter_model_open_source
        if inserter_model_open_source:  # 如果模型开源
            if inserter_model_path:
                self.inserter_model_root = inserter_model_path
            else:
                self.inserter_model_root = inserter_model_name
            self.inserter_model = None  # todo: load model
        else:
            self.inserter_model_key = inserter_model_key
            self.inserter_model_base_url = inserter_model_base_url

        # The insertion ratio represents the percentage of the input text’s word count.
        self.insertion_ratio = insertion_ratio

        self.similarity_threshold = similarity_threshold

        self.nlp = spacy.load('en_core_web_sm')  # todo: remove

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

    def _embedder(self, text: str) -> list[float]:
        """
        The EMBEDDER needs to be capable of projecting both words and documents into a high-dimensional latent space.
        Args:
            text: the input text(a word or a sentence or a paragraph or a document)(a paragraph in this case).
        Returns:
            a high-dimensional vector of the input text.
        """
        if self.embedder_model_open_source:
            embedding = self._embedding_from_open_source_model(text)
        else:
            embedding = self._embedding_from_api(text)
        return embedding

    def _load_secret_table(self) -> list[tuple[str, list[float]]]:
        """
        加载 the secret-words tabel.
        Returns:
            The secret-words tabel (list[tuple[word: str, embedding: list[float]]])
        """
        file = open(self.secret_words_table_path, 'r')
        lines = file.readlines()
        insert_table = []
        for line in lines:
            word = line.split(':')[0]
            word_embedding = extract_floats_from_string(line)
            insert_table.append((word, word_embedding))
        return insert_table

    def _cal_insert_table(self, vector: list[float]) -> list[str]:
        """
        calculate the insert-words table from the secret-words tabel.
        Args:
            vector: embedding of the input text.
        Returns:
            The insert-words table (list[str]).
        """
        insert_table = []

        if self.secret_words_table is None:
            secret_words_table = self._load_secret_table()
        else:
            secret_words_table = self.secret_words_table

        bar = tqdm(total=len(secret_words_table))
        for line in secret_words_table:
            secret_word = line[0]
            secret_word_embedding = line[1]
            # 计算 输入的句子的嵌入 与 词汇表所有单词的嵌入 的相似度
            sim = cal_cosine_similarity(vector, secret_word_embedding)
            insert_table.append((secret_word, sim))
            bar.update(1)

        # 对列表进行排序，按照元组中的浮点数排序
        insert_table_sorted = sorted(insert_table, key=lambda x: x[1], reverse=True)
        insert_table_word = [item[0] for item in insert_table_sorted]

        return insert_table_word

    def _get_insert_table_top(self, text: str, insert_table: list[str]) -> list[str]:
        doc_text = self.nlp(text)
        punctuation_removed = [token.text for token in doc_text if not token.is_punct]  # 去除标点符号
        top_n = len(punctuation_removed) * self.insertion_ratio
        return insert_table[:round(top_n)]

    def _get_words_embedding(self, words: list[str]) -> list[list[float]]:
        words_embedding = []
        if self.secret_words_table is None:
            secret_words_table = self._load_secret_table()
        else:
            secret_words_table = self.secret_words_table

        bar = tqdm(total=len(words))
        for word in words:
            # 先在秘密表中找
            find_word = False
            for secret_word in secret_words_table:
                if word == secret_word[0]:
                    words_embedding.append(secret_word[1])
                    find_word = True
                    break
            # 找不到，调模型
            if not find_word:  # todo: sql
                # 解决单词重复问题，降低模型调用次数
                find_new_word = False
                for new_word in self.new_words_table_runtime:
                    if word == new_word[0]:
                        words_embedding.append(new_word[1])
                        find_new_word = True
                        break
                if not find_new_word:
                    temp_embedding = self._embedding_from_api(word)
                    words_embedding.append(temp_embedding)
                    # 将不再秘密表中的单词加入‘运行时产生的新单词表’
                    self.new_words_table_runtime.append((word, temp_embedding))
            bar.update(1)
        # self.new_words_table_runtime.clear()  # todo: 生命周期
        return words_embedding

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

    def _count_insert_words(self, text: str, insert_words: list[str]) -> float:
        """
        Count the number of insert-words in text.
        Args:
            text:
            insert_words:
        Returns:
        """
        green_words_num = 0

        # todo: text to words list

        # method 1: use ' ' split
        # text_words = text.split()

        # method 2: spacy
        doc_text = self.nlp(text)
        text_words = [token.text for token in doc_text]  # 去除标点符号

        insert_words_embedding = self._get_words_embedding(insert_words)
        text_words_embedding = self._get_words_embedding(text_words)

        for twe in text_words_embedding:
            for swe in insert_words_embedding:
                word_sim = cal_cosine_similarity(twe, swe)
                if word_sim > self.similarity_threshold:
                    green_words_num += 1

        return green_words_num / len(insert_words)

    def watermark_text_generator(self, text: str) -> Dict[str, Any]:
        vector = self._embedder(text)
        insert_table = self._cal_insert_table(vector)
        insert_table_top = self._get_insert_table_top(text, insert_table)
        watermarked_text = self._inserter(text, insert_table_top)
        generator_result = {
            "watermarked_text": watermarked_text,
            "embedding_words": insert_table_top,
            # "original_text_embedding": vector,  # todo: sql
        }
        return generator_result

    def watermark_text_detector(self, text: str) -> Dict[str, Any]:
        vector = self._embedder(text)
        insert_table = self._cal_insert_table(vector)
        insert_table_top = self._get_insert_table_top(text, insert_table)
        p = self._count_insert_words(text, insert_table_top)

        if p > 0.3:
            watermarked = True
        else:
            watermarked = False

        detector_result = {
            "watermarked": watermarked,
            "p_value": p,
            "embedding_words": insert_table_top,
            # "watermarked_text_embedding": vector,  # todo: sql
        }
        return detector_result

# if __name__ == '__main__':
#     vec1 = [-0.015056877, -0.017690022, -0.018006723, 0.018766806, -0.0043817866]
#     vec2 = [0.015056877, 0.017690022, 0.018006723, -0.018766806, 0.0043817866]
#
#     # 计算余弦相似度
#     similarity = cal_cosine_similarity(vec1, vec2)
#     print(similarity)
