import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import torch
import torch.nn.functional as F
from torch import nn
import hashlib
from scipy.stats import norm
import gensim
import pdb
from transformers import BertForMaskedLM as WoBertForMaskedLM
# from wobert import WoBertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import BertForMaskedLM, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import gensim.downloader as api
import Levenshtein
import string
import spacy
# import paddle
from jieba import posseg

from typing import Dict, Any
from gensim.models import KeyedVectors
from models.wm_model import WatermarkModelForExistingText

# paddle.enable_static()
import re


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub('([。！？\?][”’])([^，。！？\?\n ])', r'\1\n\2', para)
    para = re.sub('(\.{6}|\…{2})([^”’\n])', r'\1\n\2', para)
    para = re.sub('([^。！？\?]*)([:：][^。！？\?\n]*)', r'\1\n\2', para)
    para = re.sub('([。！？\?][”’])$', r'\1\n', para)
    para = para.rstrip()
    return para.split("\n")


def is_subword(token: str):
    return token.startswith('##')


def binary_encoding_function(token):
    hash_value = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16)
    random_bit = hash_value % 2
    return random_bit


def is_similar(x, y, threshold=0.5):
    distance = Levenshtein.distance(x, y)
    if distance / max(len(x), len(y)) < threshold:
        return True
    return False


class WtgbblModel(WatermarkModelForExistingText):
    def __init__(
            self, substitute_model_root: str, relatedness_model_root: str, w2v_model_root: str,
            detect_mode: str = 'precise',
            tau_word: float = 0.8, tau_sent: float = 0.8, lamda: float = 0.83,
            language: str = 'en', watermark_message_type: str = 'zero-bit',
            use_z_test: bool = True, z_test_alpha: float = 0.05
    ):
        """
        Args:
            substitute_model_root: 生成替换词、计算单词的上下文嵌入相似度
            relatedness_model_root: 计算句子相似度
            w2v_model_root: word to vector model
            language: text language (default english)
            detect_mode: [fast, precise]
            tau_word: word-level similarity thresh
            tau_sent:
            lamda: word-level similarity weight
            watermark_message_type:
            use_z_test:
            z_test_alpha: 显著性水平
        """

        super().__init__(language, watermark_message_type, use_z_test, z_test_alpha)

        self.detect_mode = detect_mode
        self.tau_word = tau_word
        self.tau_sent = tau_sent
        self.lamda = lamda
        self.cn_tag_black_list = {
            '',  # 空字符串
            'x',  # 非语素字
            'u',  # 助词
            'j',  # 简称略语
            'k',  # 后接成分
            'zg',  # 状态词语素
            'y',  # 语气词
            'eng',  # 英文字符
            'uv',  # 虚拟谓词
            'uj',  # 助词
            'ud',  # 结构助词
            'nr',  # 人名
            'nrfg',  # 人名
            'nrt',  # 人名
            'nw',  # 作品名
            'nz',  # 其他专名
            'ns',  # 地名
            'nt',  # 机构团体
            'm',  # 数词
            'mq',  # 数词
            'r',  # 代词
            'w',  # 标点符号
            'PER',  # 个人，指代人物的名称或称谓
            'LOC',  # 地点，指代地理位置或地点的名称
            'ORG'  # 组织，指代公司、机构或团体的名称
        }  # set(['','f','u','nr','nw','nz','m','r','p','c','w','PER','LOC','ORG'])
        self.en_tag_white_list = {
            'MD',  # 情态动词（Modal）
            'NN',  # 名词（Noun，单数形式）
            'NNS',  # 名词（Noun，复数形式）
            'UH',  # 感叹词（Interjection）
            'VB',  # 动词（Verb，基本形式）
            'VBD',  # 动词（Verb，过去式）
            'VBG',  # 动词（Verb，现在分词）
            'VBN',  # 动词（Verb，过去分词）
            'VBP',  # 动词（Verb，非第三人称单数）
            'VBZ',  # 动词（Verb，第三人称单数）
            'RP',  # 介词副词（Particle）
            'RB',  # 副词（Adverb）
            'RBR',  # 副词（Adverb，比较级）
            'RBS',  # 副词（Adverb，最高级）
            'JJ',  # 形容词（Adjective）
            'JJR',  # 形容词（Adjective，比较级）
            'JJS'  # 形容词（Adjective，最高级）
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.language in ['chinese', 'zh']:
            self.relatedness_tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-Roberta-330M-Similarity")
            self.relatedness_model = AutoModelForSequenceClassification.from_pretrained(
                "IDEA-CCNL/Erlangshen-Roberta-330M-Similarity"
            ).to(self.device)

            # self.tokenizer = WoBertTokenizer.from_pretrained("junnyu/wobert_chinese_plus_base")
            # self.model = WoBertForMaskedLM.from_pretrained(
            #     "junnyu/wobert_chinese_plus_base",
            #     output_hidden_states=True
            # ).to(self.device)

            self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                'sgns.merge.word.bz2', binary=False, unicode_errors='ignore', limit=50000
            )

        elif self.language in ['english', 'en']:
            self.relatedness_model = RobertaForSequenceClassification.from_pretrained(
                relatedness_model_root
            ).to(self.device)
            self.relatedness_tokenizer = RobertaTokenizer.from_pretrained(relatedness_model_root)

            self.tokenizer = BertTokenizer.from_pretrained(substitute_model_root)
            self.model = BertForMaskedLM.from_pretrained(
                substitute_model_root, output_hidden_states=True
            ).to(self.device)

            # self.w2v_model = api.load("glove-wiki-gigaword-100")
            self.w2v_model = KeyedVectors.load_word2vec_format(w2v_model_root, binary=False)

            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            self.nlp = spacy.load('en_core_web_sm')

    def cut(self, ori_text, text_len):
        if self.language in ['chinese', 'zh']:
            if len(ori_text) > text_len + 5:
                ori_text = ori_text[:text_len + 5]
            if len(ori_text) < text_len - 5:
                return 'Short'
            return ori_text
        elif self.language in ['english', 'en']:
            tokens = self.tokenizer.tokenize(ori_text)
            if len(tokens) > text_len + 5:
                ori_text = self.tokenizer.convert_tokens_to_string(tokens[:text_len + 5])
            if len(tokens) < text_len - 5:
                return 'Short'
            return ori_text
        else:
            print(f'Unsupported Language:{self.language}')
            raise NotImplementedError

    def sent_tokenize(self, ori_text):
        if self.language in ['chinese', 'zh']:
            return cut_sent(ori_text)
        elif self.language in ['english', 'en']:
            return nltk.sent_tokenize(ori_text)

    def pos_filter(self, tokens, masked_token_index, input_text):
        if self.language in ['chinese', 'zh']:
            pairs = posseg.lcut(input_text)
            pos_dict = {word: pos for word, pos in pairs}
            pos_list_input = [pos for _, pos in pairs]
            pos = pos_dict.get(tokens[masked_token_index], '')
            if pos in self.cn_tag_black_list:
                return False
            else:
                return True
        elif self.language in ['english', 'en']:
            pos_tags = pos_tag(tokens)
            pos = pos_tags[masked_token_index][1]
            if pos not in self.en_tag_white_list:
                return False
            if is_subword(tokens[masked_token_index]) or is_subword(tokens[masked_token_index + 1]) or (
                    tokens[masked_token_index] in self.stop_words or tokens[masked_token_index] in string.punctuation):
                return False
            return True

    def filter_special_candidate(self, top_n_tokens, tokens, masked_token_index, input_text):
        if self.language in ['english', 'en']:
            filtered_tokens = [
                tok for tok in top_n_tokens if
                tok not in self.stop_words
                and tok not in string.punctuation
                and pos_tag([tok])[0][1] in self.en_tag_white_list
                and not is_subword(tok)
            ]
            base_word = tokens[masked_token_index]

            processed_tokens = [tok for tok in filtered_tokens if not is_similar(tok, base_word)]
            return processed_tokens

        elif self.language in ['chinese', 'zh']:
            pairs = posseg.lcut(input_text)
            pos_dict = {word: pos for word, pos in pairs}
            pos_list_input = [pos for _, pos in pairs]
            pos = pos_dict.get(tokens[masked_token_index], '')
            filtered_tokens = []

            for tok in top_n_tokens:
                watermarked_text_segtest = self.tokenizer.convert_tokens_to_string(
                    tokens[1:masked_token_index] + [tok] + tokens[masked_token_index + 1:-1]
                )
                watermarked_text_segtest = re.sub(
                    r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])',
                    '', watermarked_text_segtest
                )
                pairs_tok = posseg.lcut(watermarked_text_segtest)
                pos_dict_tok = {word: pos for word, pos in pairs_tok}
                flag = pos_dict_tok.get(tok, '')

                if flag not in self.cn_tag_black_list and flag == pos:
                    filtered_tokens.append(tok)

            processed_tokens = filtered_tokens
            return processed_tokens

    def global_word_sim(self, word, ori_word):
        try:
            global_score = self.w2v_model.similarity(word, ori_word)
        except KeyError:
            global_score = 0
        return global_score

    def context_word_sim(self, init_candidates_list, tokens, index_space, input_text):
        original_input_tensor = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        all_cos_sims = []

        for init_candidates, masked_token_index in zip(init_candidates_list, index_space):
            batch_input_ids = [[
                self.tokenizer.convert_tokens_to_ids(
                    ['[CLS]'] + tokens[1:masked_token_index] + [token] + tokens[masked_token_index + 1:-1] + ['[SEP]']
                )] for token in init_candidates
            ]
            batch_input_tensors = torch.tensor(batch_input_ids).squeeze(1).to(self.device)
            batch_input_tensors = torch.cat((batch_input_tensors, original_input_tensor), dim=0)

            with torch.no_grad():
                outputs = self.model(batch_input_tensors)
                cos_sims = torch.zeros([len(init_candidates)]).to(self.device)
                num_layers = len(outputs[1])
                N = 8
                i = masked_token_index
                # We want to calculate similarity for the last N layers
                hidden_states = outputs[1][-N:]

                # Shape of hidden_states: [N, batch_size, sequence_length, hidden_size]
                hidden_states = torch.stack(hidden_states)

                # Separate the source and candidate hidden states
                source_hidden_states = hidden_states[:, len(init_candidates):, i, :]
                candidate_hidden_states = hidden_states[:, :len(init_candidates), i, :]

                # Calculate cosine similarities across all layers and sum
                cos_sim_sum = F.cosine_similarity(
                    source_hidden_states.unsqueeze(2), candidate_hidden_states.unsqueeze(1), dim=-1
                ).sum(dim=0)

                cos_sim_avg = cos_sim_sum / N
                cos_sims += cos_sim_avg.squeeze()

            all_cos_sims.append(cos_sims.tolist())

        return all_cos_sims

    def sentence_sim(self, init_candidates_list, tokens, index_space, input_text):
        batch_size = 128
        all_batch_sentences = []
        all_index_lengths = []

        for init_candidates, masked_token_index in zip(init_candidates_list, index_space):
            if self.language in ['chinese', 'zh']:
                batch_sents = [self.tokenizer.convert_tokens_to_string(
                    tokens[1:masked_token_index] + [token] + tokens[masked_token_index + 1:-1]
                ) for token in init_candidates]
                batch_sentences = [re.sub(
                    r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])',
                    '', sent
                ) for sent in batch_sents]

                all_batch_sentences.extend([input_text + '[SEP]' + s for s in batch_sentences])

            elif self.language in ['english', 'en']:
                batch_sentences = [self.tokenizer.convert_tokens_to_string(
                    tokens[1:masked_token_index] + [token] + tokens[masked_token_index + 1:-1]
                ) for token in init_candidates]

                all_batch_sentences.extend([input_text + '</s></s>' + s for s in batch_sentences])

            all_index_lengths.append(len(init_candidates))

        all_relatedness_scores = []
        start_index = 0
        for i in range(0, len(all_batch_sentences), batch_size):
            batch_sentences = all_batch_sentences[i: i + batch_size]
            encoded_dict = self.relatedness_tokenizer.batch_encode_plus(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            input_ids = encoded_dict['input_ids'].to(self.device)
            attention_masks = encoded_dict['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.relatedness_model(input_ids=input_ids, attention_mask=attention_masks)
                logits = outputs[0]

            probs = torch.softmax(logits, dim=1)

            if self.language in ['chinese', 'zh']:
                relatedness_scores = probs[:, 1]  # .tolist()
            else:  # self.language in ['english', 'en']:
                relatedness_scores = probs[:, 2]  # .tolist()

            all_relatedness_scores.extend(relatedness_scores)

        all_relatedness_scores_split = []
        for length in all_index_lengths:
            all_relatedness_scores_split.append(all_relatedness_scores[start_index:start_index + length])
            start_index += length

        return all_relatedness_scores_split

    def candidates_gen(self, tokens, index_space, input_text, topk=64, dropout_prob=0.3):
        input_ids_bert = self.tokenizer.convert_tokens_to_ids(tokens)
        new_index_space = []
        masked_text = self.tokenizer.convert_tokens_to_string(tokens)
        # Create a tensor of input IDs
        input_tensor = torch.tensor([input_ids_bert]).to(self.device)

        with torch.no_grad():
            embeddings = self.model.bert.embeddings(input_tensor.repeat(len(index_space), 1))

        dropout = nn.Dropout2d(p=dropout_prob)

        masked_indices = torch.tensor(index_space).to(self.device)
        embeddings[torch.arange(len(index_space)), masked_indices] = dropout(
            embeddings[torch.arange(len(index_space)), masked_indices])

        with torch.no_grad():
            outputs = self.model(inputs_embeds=embeddings)

        all_processed_tokens = []
        for i, masked_token_index in enumerate(index_space):
            predicted_logits = outputs[0][i][masked_token_index]
            # Set the number of top predictions to return
            n = topk
            # Get the top n predicted tokens and their probabilities
            probs = torch.nn.functional.softmax(predicted_logits, dim=-1)
            top_n_probs, top_n_indices = torch.topk(probs, n)
            top_n_tokens = self.tokenizer.convert_ids_to_tokens(top_n_indices.tolist())
            processed_tokens = self.filter_special_candidate(top_n_tokens, tokens, masked_token_index, input_text)

            if tokens[masked_token_index] not in processed_tokens:
                processed_tokens = [tokens[masked_token_index]] + processed_tokens
            all_processed_tokens.append(processed_tokens)
            new_index_space.append(masked_token_index)

        return all_processed_tokens, new_index_space

    def filter_candidates(self, init_candidates_list, tokens, index_space, input_text):
        # 计算单词的上下文嵌入相似度
        all_context_word_similarity_scores = self.context_word_sim(
            init_candidates_list, tokens, index_space, input_text
        )
        # 计算句子相似度
        all_sentence_similarity_scores = self.sentence_sim(init_candidates_list, tokens, index_space, input_text)

        all_filtered_candidates = []
        new_index_space = []

        for init_candidates, word_context_similarity_scores, sentence_similarity_scores, masked_token_index in zip(
                init_candidates_list, all_context_word_similarity_scores, all_sentence_similarity_scores, index_space
        ):
            filtered_candidates = []
            for idx, candidate in enumerate(init_candidates):
                # 计算单词的全局词嵌入相似度
                word_global_similarity_score = self.global_word_sim(tokens[masked_token_index], candidate)
                # 综合单词的上下文嵌入和全局嵌入
                word_similarity_score = (self.lamda * word_context_similarity_scores[idx]
                                         + (1 - self.lamda) * word_global_similarity_score)

                # 阈值过滤
                if (word_similarity_score >= self.tau_word
                        and sentence_similarity_scores[idx] >= self.tau_sent
                ):
                    filtered_candidates.append((candidate, word_similarity_score))

            if len(filtered_candidates) >= 1:
                all_filtered_candidates.append(filtered_candidates)
                new_index_space.append(masked_token_index)

        return all_filtered_candidates, new_index_space

    def get_candidate_encodings(self, tokens, enhanced_candidates, index_space):
        best_candidates = []
        new_index_space = []

        for init_candidates, masked_token_index in zip(enhanced_candidates, index_space):
            filtered_candidates = []
            for idx, candidate in enumerate(init_candidates):
                if masked_token_index - 1 in new_index_space:
                    bit = binary_encoding_function(best_candidates[-1] + candidate[0])
                else:
                    bit = binary_encoding_function(tokens[masked_token_index - 1] + candidate[0])

                if bit == 1:
                    filtered_candidates.append(candidate)

            # Sort the candidates based on their scores
            filtered_candidates = sorted(filtered_candidates, key=lambda x: x[1], reverse=True)

            if len(filtered_candidates) >= 1:
                best_candidates.append(filtered_candidates[0][0])
                new_index_space.append(masked_token_index)

        return best_candidates, new_index_space

    def watermark_embed(self, text):
        input_text = text
        # Tokenize the input text
        tokens = self.tokenizer.tokenize(input_text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        masked_tokens = tokens.copy()
        start_index = 1
        end_index = len(tokens) - 1

        index_space = []
        for masked_token_index in range(start_index + 1, end_index - 1):
            binary_encoding = binary_encoding_function(tokens[masked_token_index - 1] + tokens[masked_token_index])
            if binary_encoding == 1 and masked_token_index - 1 not in index_space:
                continue
            if not self.pos_filter(tokens, masked_token_index, input_text):
                continue
            index_space.append(masked_token_index)

        if len(index_space) == 0:
            return text

        init_candidates, new_index_space = self.candidates_gen(
            tokens, index_space, input_text, 8, 0
        )

        if len(new_index_space) == 0:
            return text

        enhanced_candidates, new_index_space = self.filter_candidates(
            init_candidates, tokens, new_index_space, input_text
        )

        enhanced_candidates_2, new_index_space_2 = self.get_candidate_encodings(
            tokens, enhanced_candidates, new_index_space
        )

        for init_candidate, masked_token_index in zip(enhanced_candidates_2, new_index_space_2):
            tokens[masked_token_index] = init_candidate

        watermarked_text = self.tokenizer.convert_tokens_to_string(tokens[1:-1])

        if self.language in ['chinese', 'zh']:
            watermarked_text = re.sub(
                r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])',
                '', watermarked_text
            )

        # return watermarked_text  # todo: v1.0
        return watermarked_text, enhanced_candidates_2, new_index_space_2

    def watermark_text_generator(self, text: str) -> Dict[str, Any]:
        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)

        watermarked_text = ''
        # for i in range(0, num_sents, 2):  # todo: v1.0
        #     if i + 1 < num_sents:
        #         sent_pair = sents[i] + sents[i + 1]
        #     else:
        #         sent_pair = sents[i]
        #     # keywords = jieba.analyse.extract_tags(sent_pair, topK=5, withWeight=False)
        #     if len(watermarked_text) == 0:
        #         watermarked_text = self.watermark_embed(sent_pair)
        #     else:
        #         watermarked_text = watermarked_text + self.watermark_embed(sent_pair)

        all_candidates, all_index = [], []
        for i in range(num_sents):
            temp_text, temp_candidates, temp_i = self.watermark_embed(sents[i])
            watermarked_text = watermarked_text + temp_text + ' '
            all_candidates.append(temp_candidates)
            all_index.append(temp_i)

        if len(self.get_encodings_fast(text)) == 0:
            # print(text)
            watermarked_text = ''

        generator_result = {
            "watermarked_text": watermarked_text.strip(),  # 去掉最后一个空格
            "message_words": all_candidates,
            "message_words_i": all_index
        }
        return generator_result

    def get_encodings_fast(self, text):
        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)

        encodings = []
        for i in range(0, num_sents, 2):
            if i + 1 < num_sents:
                sent_pair = sents[i] + sents[i + 1]
            else:
                sent_pair = sents[i]
            tokens = self.tokenizer.tokenize(sent_pair)

            for index in range(1, len(tokens) - 1):
                if not self.pos_filter(tokens, index, text):
                    continue
                bit = binary_encoding_function(tokens[index - 1] + tokens[index])
                encodings.append(bit)

        return encodings

    def get_encodings_precise(self, text):
        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)

        encodings = []
        for i in range(0, num_sents, 2):
            if i + 1 < num_sents:
                sent_pair = sents[i] + sents[i + 1]
            else:
                sent_pair = sents[i]

            tokens = self.tokenizer.tokenize(sent_pair)
            tokens = ['[CLS]'] + tokens + ['[SEP]']

            masked_tokens = tokens.copy()

            start_index = 1
            end_index = len(tokens) - 1

            index_space = []
            for masked_token_index in range(start_index + 1, end_index - 1):
                if not self.pos_filter(tokens, masked_token_index, sent_pair):
                    continue
                index_space.append(masked_token_index)

            if len(index_space) == 0:
                continue

            init_candidates, new_index_space = self.candidates_gen(
                tokens, index_space, sent_pair, 8, 0
            )
            enhanced_candidates, new_index_space = self.filter_candidates(
                init_candidates, tokens, new_index_space, sent_pair
            )

            for j, idx in enumerate(new_index_space):
                if len(enhanced_candidates[j]) > 1:
                    bit = binary_encoding_function(tokens[idx - 1] + tokens[idx])
                    encodings.append(bit)

        return encodings

    def watermark_text_detector(self, text: str, alpha=0.05) -> Dict[str, Any]:
        if self.detect_mode == 'precise':
            encodings = self.get_encodings_precise(text)
        else:  # self.detect_mode == 'fast'
            encodings = self.get_encodings_fast(text)
        n = len(encodings)
        ones = sum(encodings)
        if n == 0:
            z_score = 0
        else:
            z_score = (ones - self.p * n) / (n * self.p * (1 - self.p)) ** 0.5
        p_value = norm.sf(z_score)
        confidence = (1 - p_value) * 100
        is_watermark = z_score >= self.z_alpha

        detector_result = {
            "watermarked": is_watermark,
            "p_value": p_value,
            "confidence": confidence,
            "encoding_length": n,
            "encoding_ones": ones,
            "z_score": z_score,
        }
        return detector_result
