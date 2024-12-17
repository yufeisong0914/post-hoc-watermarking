# parals
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fairseq.models.transformer import TransformerModel

# black_box
import copy

import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from jieba import posseg

import torch
import torch.nn.functional as F
from torch import nn

# from transformers import BertForMaskedLM as WoBertForMaskedLM
# from wobert import WoBertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForMaskedLM, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer

import hashlib
import spacy
from scipy.stats import norm
from nltk.tokenize import sent_tokenize
import gensim
from gensim.models import KeyedVectors
import Levenshtein
import string

import re
from models.keyword_extractor import KeywordExtractor
from models.mask_selector import MaskSelector
from ciwater_extrect_parals import lexical_substitute_transformer_model
import models.util_string as util_string

w2v_model_home = "/home/haojifei/develop_tools/w2v_models"


def preprocess_txt(corpus):
    corpus = [t.replace("\n", " ") for t in corpus]
    corpus = [t.replace("\t", " ") for t in corpus]
    # html tag
    CLEANR = re.compile('<.*?>')
    corpus = [re.sub(CLEANR, '', c) for c in corpus if len(c) > 0]
    corpus = [re.sub(r"\.+", ".", c) for c in corpus]
    return corpus


def cut_sent(para):
    """
    切分中文句子
    Args:
        para:
    Returns:
    """
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


class watermark_model:
    def __init__(
            self, models_dir: str, w2v_dir: str,
            language: str, detect_mode: str, alpha: float,
            tau_word: float, tau_sent: float, lamda: float
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.detect_mode = detect_mode
        self.alpha = alpha
        self.z_alpha = norm.ppf(1 - self.alpha, loc=0, scale=1)
        self.p = 0.5
        self.tau_word = tau_word
        self.tau_sent = tau_sent
        self.lamda = lamda
        # 中文--jieba分词词性
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
        }  # {'','f','u','nr','nw','nz','m','r','p','c','w','PER','LOC','ORG'}
        # 英文--nltk分词词性
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

        if language == 'Chinese':
            self.relatedness_tokenizer = AutoTokenizer.from_pretrained(
                models_dir + "/IDEA-CCNL/Erlangshen-Roberta-330M-Similarity"
            )
            self.relatedness_model = AutoModelForSequenceClassification.from_pretrained(
                models_dir + "/IDEA-CCNL/Erlangshen-Roberta-330M-Similarity"
            ).to(self.device)

            # self.tokenizer = WoBertTokenizer.from_pretrained(models_dir + "/junnyu/wobert_chinese_plus_base")
            # self.model = WoBertForMaskedLM.from_pretrained(
            #     models_dir + "/junnyu/wobert_chinese_plus_base",
            #     output_hidden_states=True
            # ).to(self.device)

            self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                w2v_model_home + '/sgns.merge.word.bz2',
                binary=False,
                unicode_errors='ignore',
                limit=50000
            )

        elif language == 'English':
            # self.relatedness_tokenizer = RobertaTokenizer.from_pretrained(
            #     models_dir + '/FacebookAI/roberta-large-mnli'
            # )
            # self.relatedness_model = RobertaForSequenceClassification.from_pretrained(
            #     models_dir + '/FacebookAI/roberta-large-mnli'
            # ).to(self.device)
            #
            # self.tokenizer = BertTokenizer.from_pretrained(models_dir + '/google-bert/bert-base-cased')
            # self.model = BertForMaskedLM.from_pretrained(
            #     models_dir + '/google-bert/bert-base-cased',
            #     output_hidden_states=True
            # ).to(self.device)

            self.model = TransformerModel.from_pretrained(
            '/home/haojifei/develop_tools/my_models/parals-checkpoints/para/transformer/',
            'checkpoint_best.pt',
            bpe='subword_nmt',
            bpe_codes='/home/haojifei/develop_tools/my_models/parals-checkpoints/para/transformer/codes.40000.bpe.en'
        ).cuda().eval()
            self.tokenizer = self.model

            self.w2v_model = KeyedVectors.load_word2vec_format(
                w2v_model_home + '/glove-wiki-gigaword-100.word2vec.txt',
                binary=False
            )

            # nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            self.nlp = spacy.load('en_core_web_sm')

        self.keyword_extractor = KeywordExtractor()
        self.mask_selector = MaskSelector(
            custom_keywords=["haojifei", "ciwater"], method="grammar", mask_order_by="dep", exclude_cc=False
        )
        self.keyword_extract_rate = 0.5
        self.replacement_rate = 1.0

    def paragraph_tokenize(self, paragraph):
        """
        将一段话分割为句子列表
        Args:
            paragraph: 一段话 "paragraph"
        Returns: 句子列表 ["sentence1", "sentence2", ...]
        """
        if self.language == 'Chinese':
            return cut_sent(paragraph)
        elif self.language == 'English':
            paragraph = util_string.preprocess_string(paragraph)  # 预处理一下
            # todo: nltk
            return sent_tokenize(paragraph)
            # todo: spacy
            # doc_paragraph = self.nlp(paragraph)
            # sentences = [sentence.text for sentence in doc_paragraph.sents]
            # return [s for s in sentences if s.strip()]


    def embed(self, paragraph):
        sentences = self.paragraph_tokenize(paragraph)
        num_sentences = len(sentences)

        watermarked_text = ""
        for i in range(0, num_sentences, 2):  # 一次处理两个句子
            if i + 1 < num_sentences:
                sent_pair = sentences[i] + " " + sentences[i + 1]
            else:
                sent_pair = sentences[i]
            watermarked_text = watermarked_text + self.watermark_embed(sent_pair) + " "

        watermarked_text = util_string.preprocess_string(watermarked_text)
        return watermarked_text

    def embed_v1(self, ori_paragraph):
        sentences = self.paragraph_tokenize(ori_paragraph)

        q = 0  # todo:delete

        watermarked_text = ""
        all_wm_index = []
        for text in sentences:
            sent_pair_wmd, wm_index = self.watermark_embed_v1(text)
            watermarked_text = watermarked_text + sent_pair_wmd + " "
            all_wm_index.append(wm_index)  # dug的时候看看
            q = q + len(wm_index)  # todo:delete

        print(q)  # todo:delete

        watermarked_text = util_string.preprocess_string(watermarked_text)
        # return watermarked_text
        return watermarked_text, all_wm_index

    def watermark_embed(self, text):
        input_text = text
        # Tokenize the input text
        tokens = self.tokenizer.tokenize(input_text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        # masked_tokens = tokens.copy()
        start_index = 2  # 1
        end_index = len(tokens) - 2  # -1

        # 存放所有需要替换的token的下标
        index_space = []
        # 根据当前词和前一个词生成当前词的 bit = [0 or 1], 如果是0要把下标加到列表index_space中
        for masked_token_index in range(start_index, end_index):  # +1，-1
            binary_encoding = binary_encoding_function(tokens[masked_token_index - 1] + tokens[masked_token_index])
            print(tokens[masked_token_index], binary_encoding)
            if binary_encoding == 1 and masked_token_index - 1 not in index_space:
                continue
            if not self.pos_filter(tokens, masked_token_index, input_text):
                print(tokens[masked_token_index], "pos!")
                continue
            index_space.append(masked_token_index)

        if len(index_space) == 0:
            return text

        print(index_space)

        # 候选词的生成
        init_candidates, new_index_space = self.candidates_gen(tokens, index_space, input_text, 8, 0)

        if len(new_index_space) == 0:
            return text

        # 评分，过滤
        enhanced_candidates, new_index_space = self.filter_candidates(
            init_candidates,
            tokens,
            new_index_space,
            input_text
        )
        enhanced_candidates, new_index_space = self.get_candidate_encodings(
            tokens,
            enhanced_candidates,
            new_index_space
        )

        for init_candidate, masked_token_index in zip(enhanced_candidates, new_index_space):
            tokens[masked_token_index] = init_candidate
        watermarked_text = self.tokenizer.convert_tokens_to_string(tokens[1:-1])

        if self.language == 'Chinese':
            watermarked_text = re.sub(
                r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])',
                '',
                watermarked_text
            )
        return watermarked_text

    def watermark_embed_v1(self, text):
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

        return text, mask_words_index_new  # todo:delete

        # 为每一个maskword生成候选词
        all_candidates = []
        for i in mask_words_index_new:
            _, candidates = self.candidates_gen_v1(doc_sentence, i)
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

    def candidates_gen_v1(self, doc_sentence, mask_word_index):
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

    def filter_candidates(self, init_candidates_list, tokens, index_space, input_text):

        all_context_word_similarity_scores = self.context_word_sim(
            init_candidates_list,
            tokens,
            index_space,
            input_text
        )

        all_sentence_similarity_scores = self.sentence_sim(init_candidates_list, tokens, index_space, input_text)

        all_filtered_candidates = []
        new_index_space = []

        for init_candidates, context_word_similarity_scores, sentence_similarity_scores, masked_token_index in zip(
                init_candidates_list, all_context_word_similarity_scores, all_sentence_similarity_scores, index_space
        ):
            filtered_candidates = []
            for idx, candidate in enumerate(init_candidates):
                global_word_similarity_score = self.global_word_sim(tokens[masked_token_index], candidate)
                word_similarity_score = self.lamda * context_word_similarity_scores[idx] + (
                        1 - self.lamda) * global_word_similarity_score
                if word_similarity_score >= self.tau_word and sentence_similarity_scores[idx] >= self.tau_sent:
                    filtered_candidates.append((candidate, word_similarity_score))

            if len(filtered_candidates) >= 1:
                all_filtered_candidates.append(filtered_candidates)
                new_index_space.append(masked_token_index)

        return all_filtered_candidates, new_index_space

    def token_detectable_filter(self, good_candidates_index, candidates, doc_sentence, i):
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

    def pos_filter(self, tokens, masked_token_index, input_text):
        if self.language == 'Chinese':
            pairs = posseg.lcut(input_text)
            pos_dict = {word: pos for word, pos in pairs}
            pos_list_input = [pos for _, pos in pairs]
            pos = pos_dict.get(tokens[masked_token_index], '')
            if pos in self.cn_tag_black_list:
                return False
            else:
                return True
        elif self.language == 'English':
            pos_tags = pos_tag(tokens)
            pos = pos_tags[masked_token_index][1]
            if pos not in self.en_tag_white_list:
                return False
            if (is_subword(tokens[masked_token_index])
                    or is_subword(tokens[masked_token_index + 1])
                    or tokens[masked_token_index] in self.stop_words
                    or tokens[masked_token_index] in string.punctuation
            ):
                return False
            return True

    def filter_special_candidate(self, top_n_tokens, tokens, masked_token_index, input_text):
        if self.language == 'English':
            filtered_tokens = [tok for tok in top_n_tokens if
                               tok not in self.stop_words and tok not in string.punctuation and pos_tag([tok])[0][
                                   1] in self.en_tag_white_list and not is_subword(tok)]

            base_word = tokens[masked_token_index]

            processed_tokens = [tok for tok in filtered_tokens if not is_similar(tok, base_word)]
            return processed_tokens
        elif self.language == 'Chinese':
            pairs = posseg.lcut(input_text)
            pos_dict = {word: pos for word, pos in pairs}
            pos_list_input = [pos for _, pos in pairs]
            pos = pos_dict.get(tokens[masked_token_index], '')
            filtered_tokens = []
            for tok in top_n_tokens:
                watermarked_text_segtest = self.tokenizer.convert_tokens_to_string(
                    tokens[1:masked_token_index] + [tok] + tokens[masked_token_index + 1:-1])
                watermarked_text_segtest = re.sub(
                    r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])',
                    '', watermarked_text_segtest)
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
            batch_input_ids = [
                [self.tokenizer.convert_tokens_to_ids(
                    ['[CLS]'] + tokens[1:masked_token_index] + [token] + tokens[masked_token_index + 1:-1] + ['[SEP]'])]
                for token in
                init_candidates]
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
                cos_sim_sum = F.cosine_similarity(source_hidden_states.unsqueeze(2),
                                                  candidate_hidden_states.unsqueeze(1), dim=-1).sum(dim=0)

                cos_sim_avg = cos_sim_sum / N
                cos_sims += cos_sim_avg.squeeze()

            all_cos_sims.append(cos_sims.tolist())

        return all_cos_sims

    def sentence_sim(self, init_candidates_list, tokens, index_space, input_text):
        batch_size = 128
        all_batch_sentences = []
        all_index_lengths = []
        for init_candidates, masked_token_index in zip(init_candidates_list, index_space):
            if self.language == 'Chinese':
                batch_sents = [self.tokenizer.convert_tokens_to_string(
                    tokens[1:masked_token_index] + [token] + tokens[masked_token_index + 1:-1]) for token in
                    init_candidates]
                batch_sentences = [re.sub(
                    r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])',
                    '', sent) for sent in batch_sents]
                all_batch_sentences.extend([input_text + '[SEP]' + s for s in batch_sentences])

            elif self.language == 'English':
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
            if self.language == 'Chinese':
                relatedness_scores = probs[:, 1]  # .tolist()
            elif self.language == 'English':
                relatedness_scores = probs[:, 2]  # .tolist()
            all_relatedness_scores.extend(relatedness_scores)

        all_relatedness_scores_split = []
        for length in all_index_lengths:
            all_relatedness_scores_split.append(all_relatedness_scores[start_index:start_index + length])
            start_index += length

        return all_relatedness_scores_split

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

    def get_encodings_fast(self, text):
        sents = self.paragraph_tokenize(text)
        num_sentences = len(sents)
        encodings = []
        for i in range(0, num_sentences, 2):
            if i + 1 < num_sentences:
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
        sentences = self.paragraph_tokenize(text)
        num_sentences = len(sentences)
        encodings = []
        for i in range(0, num_sentences, 2):
            if i + 1 < num_sentences:
                sent_pair = sentences[i] + sentences[i + 1]
            else:
                sent_pair = sentences[i]

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

            init_candidates, new_index_space = self.candidates_gen(tokens, index_space, sent_pair, 8, 0)
            enhanced_candidates, new_index_space = self.filter_candidates(
                init_candidates, tokens, new_index_space, sent_pair
            )

            for j, idx in enumerate(new_index_space):
                if len(enhanced_candidates[j]) > 1:
                    bit = binary_encoding_function(tokens[idx - 1] + tokens[idx])
                    encodings.append(bit)

        return encodings

    def get_encodings_precise_v1(self, text):
        sentences = self.paragraph_tokenize(text)
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

    def watermark_detector(self, text):
        if self.detect_mode == "precise":
            encodings = self.get_encodings_precise(text)
            mask_words_index_list, mask_words_list = [], []
        elif self.detect_mode == "fast":
            encodings = self.get_encodings_fast(text)
            mask_words_index_list, mask_words_list = [], []
        else:
            encodings, mask_words_index_list, mask_words_list = self.get_encodings_precise_v1(text)

        n = len(encodings)
        ones = sum(encodings)
        if n == 0:
            z = 0
        else:
            z = (ones - self.p * n) / (n * self.p * (1 - self.p)) ** 0.5

        p_value = norm.sf(z)
        # p_value = norm.sf(abs(z)) * 2
        is_watermark = (z >= self.z_alpha)

        return is_watermark, p_value, n, ones, z, mask_words_index_list, mask_words_list
