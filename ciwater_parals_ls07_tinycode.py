#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import json
import numpy as np
import nltk

import torch
from torch import nn
import torch.nn.functional as F

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

from reader import Reader_lexical

from fairseq.models.bart import BARTModel
from fairseq.models.transformer import TransformerModel
from bart_score import BARTScorer

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from metrics.evaluation import evaluation

import re

transformers_source_root = "/home/haojifei/develop_tools/transformers"
bart_checkpoint = "/models/facebook/bart-large-cnn"
bleurt_checkpoint = "/models/Elron/bleurt-large-512"

bart_scorer = BARTScorer(device="cuda", checkpoint=transformers_source_root + bart_checkpoint)
bart_scorer.load(path="/home/haojifei/develop_things/nlp_projects/ParaLS-main/bart_score.pth")

bleurt_tokenizer = AutoTokenizer.from_pretrained(transformers_source_root + bleurt_checkpoint)
bleurt_scorer = AutoModelForSequenceClassification.from_pretrained(transformers_source_root + bleurt_checkpoint).cuda()
bleurt_scorer.eval()

word_pos_fp = "../../Gloss/LS_infer/vocab/word_pos.json"
with open(word_pos_fp, "r") as f:
    pos_vocab = json.loads(f.read().strip())

ps = PorterStemmer()

import string

punctuation = string.punctuation
qutos = ["<unk>", '']


def gen_gpt2_string(tgt_dict, input):
    if type(input) == int:
        input = torch.tensor([input])
    return tgt_dict.bpe.decode(tgt_dict.task.source_dictionary.string(input))


def lemma_word(target, target_pos):
    # nltk.wordnet.WordNetLemmatizer()方法是NLTK库中的一个词形还原器类，用于将单词转换为它们的基本词形。
    wordnet_lemmatizer = WordNetLemmatizer()

    to_wordnet_pos = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}
    from_lst_pos = {'j': 'J', 'a': 'J', 'v': 'V', 'n': 'N', 'r': 'R'}
    try:
        pos_initial = to_wordnet_pos[from_lst_pos[target_pos]]
    except KeyError:
        pos_initial = to_wordnet_pos[target_pos]

    return wordnet_lemmatizer.lemmatize(target, pos=pos_initial)


def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string


def give_embedding_scores(outputs, tokens_embedding, complex_tokens, temperature=None, prefix_len=None):
    complex_embed = tokens_embedding[complex_tokens[:-1]]
    outputs_embed = tokens_embedding[outputs[:, 1:]]

    sim_cal = nn.CosineSimilarity(dim=-1, eps=1e-6)
    if complex_embed.size(0) == 1:
        sim_matrix = sim_cal(outputs_embed, complex_embed)
        if temperature != None:
            sim_matrix = sim_matrix / temperature
        sim_matrix = F.log_softmax(sim_matrix, dim=0)
    else:
        sim_matrix = torch.zeros(outputs.size(0), (outputs.size(1) - 1))
    return sim_matrix


@torch.no_grad()
def cal_bart_score(sentence, complex_word, complex_index, candis):
    candis_scores = []
    sentence_list = sentence.split()
    # prefix=" ".join(sentence.split()[max(0,complex_index-4):complex_index])
    # suffix=" ".join(sentence.split()[complex_index+1:min(complex_index+5,len(sentence_list))])

    prefix = " ".join(sentence.split()[0:complex_index]).strip()
    suffix = " ".join(sentence.split()[complex_index + 1:]).strip()

    ori_sentence = prefix + " " + complex_word.strip() + " " + suffix
    ori_sentence = ori_sentence.strip()
    cal_oris = []
    cal_news = []
    for candi in candis:
        cal_oris.append(ori_sentence)
        new_sentence = prefix + " " + candi + " " + suffix
        new_sentence = new_sentence.strip()
        cal_news.append(new_sentence)

    # F1=bart_scorer.score(cal_news,cal_oris)
    with torch.no_grad():
        F1 = bart_scorer.score(cal_oris, cal_news)
    # F1=np.exp(F1).tolist()
    F1 = torch.tensor(F1)
    return F1


@torch.no_grad()
def cal_bleurt_score(sentence, complex_word, complex_index, candis):
    candis_scores = []
    sentence_list = sentence.split()

    prefix = " ".join(sentence.split()[0:complex_index]).strip()
    suffix = " ".join(sentence.split()[complex_index + 1:]).strip()

    ori_sentence = prefix + " " + complex_word.strip() + " " + suffix
    ori_sentence = ori_sentence.strip()
    cal_oris = []
    cal_news = []
    for candi in candis:
        cal_oris.append(ori_sentence)
        new_sentence = prefix + " " + candi + " " + suffix
        new_sentence = new_sentence.strip()
        cal_news.append(new_sentence)

    # F1=bart_scorer.score(cal_news,cal_oris)
    with torch.no_grad():
        # F1=bleurt_scorer.score(references=cal_oris, candidates=cal_news)
        input_dict = bleurt_tokenizer(cal_oris, cal_news, return_tensors='pt', padding=True)
        F1 = bleurt_scorer(
            input_ids=input_dict["input_ids"].cuda(),
            token_type_ids=input_dict["token_type_ids"].cuda(),
            attention_mask=input_dict["attention_mask"].cuda()
        )[0].squeeze()
    # F1=np.exp(F1).tolist()
    # import pdb
    # pdb.set_trace()
    # F1=torch.tensor(F1)
    return F1.cpu()


def give_real_scores_ahead(tgt_dict, outputs, scores_with_suffix, scores_with_suffix_masks, suffix_tokens,
                           prefix_len=None, prefix_str=None, max_ahead=1, flag=1):
    beam_size, max_len = outputs.size()
    scores_with_suffix = scores_with_suffix[:, :max_len - 1]
    scores_with_suffix_masks = scores_with_suffix_masks[:, :max_len - 1]

    first_index = prefix_len
    last_index = min(prefix_len + max_ahead, max_len - 1)
    # print(scores_with_suffix[:,0:5])
    for i in range(first_index, last_index):
        if first_index > 0:
            scores_with_suffix[:, i] -= scores_with_suffix[:, first_index - 1]
        else:
            pass
    # print(outputs)
    # print(scores_with_suffix[:,0:5])
    # for i in range(first_index,last_index):
    #     pass
    # scores_with_suffix[:,i]/=(len(suffix_tokens)+i-prefix_len+1)
    # scores_with_suffix[:,i]/=(len(suffix_tokens)+i-prefix_len+1)
    # print(scores_with_suffix[:,0:5])
    scores_with_suffix[scores_with_suffix_masks] = -math.inf
    for j in range(0, first_index):
        scores_with_suffix[:, j] = torch.tensor(-math.inf)
    for j in range(last_index, max_len - 1):
        scores_with_suffix[:, j] = torch.tensor(-math.inf)

    flat_scores_with_suffix = scores_with_suffix.reshape(1, -1).squeeze(dim=0)
    sorted_scores, sorted_indices = torch.topk(flat_scores_with_suffix, k=beam_size * (last_index - first_index))
    beam_idx = sorted_indices // (max_len - 1)
    len_idx = (sorted_indices % (max_len - 1)) + 1
    if flag != None:
        hope_len = len(nltk.word_tokenize(prefix_str)) + flag
        # hope_len=len(prefix_str.strip().split())+flag
    else:
        hope_len = -1

    hope_outputs = []
    hope_outputs_scores = []
    candis = []
    for i in range(len(beam_idx)):
        if sorted_scores[i] == (-math.inf):
            continue
        tmp_str1 = tgt_dict.string(outputs[beam_idx[i], :(len_idx[i] + 1)]).replace("@@ ", "")
        # tmp_str1=tmp_str1.replace("<unk>","")
        # if len(tmp_str1.strip().split())==hope_len:
        if "<unk>" in tmp_str1:
            print("finding a unk in prefix str")
            tmp_str1 = tmp_str1.replace("<unk>", "|")

        if len(nltk.word_tokenize(tmp_str1)) == hope_len:
            candis.append(nltk.word_tokenize(tmp_str1)[-1].strip())
            hope_outputs.append(outputs[beam_idx[i]])
            # print(tgt_dict.string(outputs[beam_idx[i]]),sorted_scores[i])
            hope_outputs_scores.append(sorted_scores[i].tolist())
            # hope_outputs_scores.append(scores_with_suffix[beam_idx[i]][first_index].tolist())

        elif hope_len == -1:
            hope_outputs.append(outputs[beam_idx[i]])
            hope_outputs_scores.append(sorted_scores[i].tolist())
        # if len(tmp_str1.split())==len(prefix_str.split())+1:
        #     print(tmp_str1)
    # print("&"*100)
    # import pdb
    # pdb.set_trace()
    return hope_outputs, hope_outputs_scores, candis


def give_real_scores_ahead_bart(
        tgt_dict, outputs, scores_with_suffix, scores_with_suffix_masks, suffix_tokens,
        prefix_len=None, prefix_str=None, max_ahead=1, flag=1
):
    beam_size, max_len = outputs.size()
    scores_with_suffix = scores_with_suffix[:, :max_len - 1]
    scores_with_suffix_masks = scores_with_suffix_masks[:, :max_len - 1]

    first_index = prefix_len
    last_index = min(prefix_len + max_ahead, max_len - 1)

    for i in range(first_index, last_index):
        if first_index > 0:
            scores_with_suffix[:, i] -= scores_with_suffix[:, first_index - 1]
        else:
            pass

    ahead_parts = outputs[:, 1:]
    ahead_parts = ahead_parts.reshape(1, -1)[0].tolist()
    # ahead_part_tokens=list(map(lambda x:tokenizer.convert_ids_to_tokens(x),ahead_parts))
    ahead_part_tokens = list(map(lambda x: gen_gpt2_string(tgt_dict, x), ahead_parts))
    ahead_part_tokens_masks = list(map(lambda x: not x.startswith(" ") and x not in qutos, ahead_part_tokens))
    ahead_part_tokens_masks = torch.tensor(ahead_part_tokens_masks).reshape(beam_size, -1)
    scores_with_suffix[:, :-1][ahead_part_tokens_masks[:, 1:]] = -math.inf

    if first_index > 1:
        ids_after_prefix = outputs[:, first_index + 1]
        ids_after_prefix = ids_after_prefix.reshape(1, -1)[0].tolist()
        ids_after_prefix_tokens = list(map(lambda x: gen_gpt2_string(tgt_dict, x), ids_after_prefix))
        ids_after_prefix_tokens_masks = list(
            map(lambda x: not x.startswith(" ") and x not in qutos, ids_after_prefix_tokens))
        ids_after_prefix_tokens_masks = torch.tensor(ids_after_prefix_tokens_masks).reshape(beam_size, -1)
        scores_with_suffix[ids_after_prefix_tokens_masks.squeeze()] = -math.inf

    scores_with_suffix[scores_with_suffix_masks] = -math.inf
    for j in range(0, first_index):
        scores_with_suffix[:, j] = torch.tensor(-math.inf)
    for j in range(last_index, max_len - 1):
        scores_with_suffix[:, j] = torch.tensor(-math.inf)

    flat_scores_with_suffix = scores_with_suffix.reshape(1, -1).squeeze(dim=0)
    sorted_scores, sorted_indices = torch.topk(flat_scores_with_suffix, k=beam_size * (last_index - first_index))
    beam_idx = sorted_indices // (max_len - 1)
    len_idx = (sorted_indices % (max_len - 1)) + 1
    if flag != None:
        hope_len = len(nltk.word_tokenize(prefix_str)) + flag
        # hope_len=len(prefix_str.strip().split())+flag
    else:
        hope_len = -1

    hope_outputs = []
    hope_outputs_scores = []
    candis = []
    for i in range(len(beam_idx)):
        if sorted_scores[i] == (-math.inf):
            continue

        tmp_str1 = tgt_dict.bpe.decode(
            tgt_dict.task.source_dictionary.string(outputs[beam_idx[i], :(len_idx[i] + 1)])).strip()

        if "<unk>" in tmp_str1:
            print("finding a unk in prefix str")
            tmp_str1 = tmp_str1.replace("<unk>", "|")

        if len(nltk.word_tokenize(tmp_str1)) == hope_len:
            # candis.append(nltk.word_tokenize(tmp_str1)[-1].strip())
            candis.append(" " + nltk.word_tokenize(tmp_str1)[-1].strip())
            hope_outputs.append(outputs[beam_idx[i]])
            # print(tgt_dict.string(outputs[beam_idx[i]]),sorted_scores[i])
            hope_outputs_scores.append(sorted_scores[i].tolist())
        elif hope_len == -1:
            hope_outputs.append(outputs[beam_idx[i]])
            hope_outputs_scores.append(sorted_scores[i].tolist())

    return hope_outputs, hope_outputs_scores, candis


def extract_substitute(output_sentences, original_sentence, complex_word, threshold, prev_scores=None, word_index=None,
                       sentence_words=None, target_pos=None, target_lemma=None):
    original_words = sentence_words
    index_of_complex_word = word_index
    if index_of_complex_word == -1:
        print("******************no found the complex word*****************")
        return [], []

    len_original_words = len(original_words)
    context = original_words[max(0, index_of_complex_word - 4):min(index_of_complex_word + 5, len_original_words)]
    context = " ".join([word for word in context])

    if index_of_complex_word < 4:
        index_of_complex_in_context = index_of_complex_word
    else:
        index_of_complex_in_context = 4

    context = (context, index_of_complex_in_context)

    if output_sentences[0].find('<unk>'):

        for i in range(len(output_sentences)):
            tran = output_sentences[i].replace('<unk>', '|')
            output_sentences[i] = tran

    complex_stem = ps.stem(complex_word)
    # orig_pos = nltk.pos_tag(original_words)[index_of_complex_word][1]
    # not_candi = {'the', 'with', 'of', 'a', 'an', 'for', 'in', "-", "``", "*", "\"","it"}
    not_candi = set(['the', 'with', 'of', 'a', 'an', 'for', 'in', "-", "``", "*", "\""])
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    not_candi.add(target_lemma)
    if len(original_words) > 1:
        not_candi.add(original_words[index_of_complex_word - 1])
    if len(original_words) > index_of_complex_word + 1:
        not_candi.add(original_words[index_of_complex_word + 1])

    # para_scores = []
    substitutes = []

    suffix_words = []

    if index_of_complex_word + 1 < len_original_words:
        suffix_words.append(original_words[
                                index_of_complex_word + 1])  # suffix_words.append(original_words[index_of_complex_word+1:min(index_of_complex_word+4,len_original_words)])
    else:
        suffix_words.append("")

    # pdb.set_trace()

    for sentence in output_sentences:

        if len(sentence) < 3:
            continue
        assert len(sentence.split()[:index_of_complex_word]) == len(original_words[:index_of_complex_word])
        words = original_words[:index_of_complex_word] + nltk.word_tokenize(
            " ".join(sentence.split()[index_of_complex_word:]))

        if index_of_complex_word >= len(words):
            continue

        if words[index_of_complex_word] == complex_word:
            len_words = len(words)
            if index_of_complex_word + 1 < len_words:
                suffix = words[
                    index_of_complex_word + 1]  # words[index_of_complex_word+1:min(index_of_complex_word+4,len_words)]
                if suffix not in suffix_words:
                    suffix_words.append(suffix)
    real_prev_scores = []
    s1_count = -1
    for sentence in output_sentences:
        s1_count += 1
        if len(sentence) < 3:
            continue
        assert len(sentence.split()[:index_of_complex_word]) == len(original_words[:index_of_complex_word])
        words = original_words[:index_of_complex_word] + nltk.word_tokenize(
            " ".join(sentence.split()[index_of_complex_word:]))

        if index_of_complex_word >= len(words):
            continue
        candi = words[index_of_complex_word].lower()
        candi_stem = ps.stem(candi)
        candi_lemma = lemma_word(candi, target_pos=target_pos)
        not_index_0 = candi.find("-")
        not_index_1 = candi.find(complex_word)
        if candi_lemma == target_lemma or candi_stem in not_candi or candi in not_candi or not_index_0 != -1 \
                or not_index_1 != -1:
            continue

        len_words = len(words)
        sent_suffix = ""
        if index_of_complex_word + 1 < len_words:
            sent_suffix = words[index_of_complex_word + 1]

        # if sent_suffix in suffix_words:
        if candi not in substitutes:
            substitutes.append(candi)
            real_prev_scores.append(prev_scores[s1_count])

    if len(substitutes) > 0:
        filter_substitutes = substitutes
        rank_bert_substitutes = substitutes

        assert len(filter_substitutes) == len(real_prev_scores)
        assert len(filter_substitutes) == len(rank_bert_substitutes)

        return filter_substitutes, rank_bert_substitutes, real_prev_scores

    return [], [], []


def lexical_substitute_transformer_model(
        model, sentence, sentence_words,
        prefix, word_index, complex_word, target_pos, target_lemma, beam, threshold
):
    index_complex = word_index  # 目标单词在句子中的索引
    ori_words = sentence_words
    prefix = prefix
    suffix1 = ""
    if index_complex != -1:
        prefix = prefix
        # 在原句子中，如果目标词后还有其他单词（即，如果目标词不是句子的最后一个词）
        if len(ori_words) > index_complex + 1:
            # 后缀
            suffix1 = " ".join(ori_words[index_complex + 1:]).strip()
            suffix1 = suffix1.replace("''", "\"").strip()
            suffix1 = suffix1.replace("``", "\"").strip()
            suffix1 = process_string(suffix1)
            # stored_suffix1=suffix1

            if suffix1.endswith("\""):
                suffix1 = suffix1[:-1]
                suffix1 = suffix1.strip()
            if suffix1.endswith("'"):
                suffix1 = suffix1[:-1]
                suffix1 = suffix1.strip()
            # 后缀只选选2个token
            suffix1 = " ".join(suffix1.split(" ")[:2])
        else:
            pass
        # print(prefix)
    else:
        print("*************cannot find the complex word")
        # print(sentence)
        # print(complex_word)
        sentence = sentence.lower()
        return lexical_substitute_transformer_model(model, sentence, complex_word, beam, threshold)

    # 将前缀编码
    prefix_tokens = model.encode(prefix)

    # [:-1]表示从序列的第一个元素开始，倒数第二个元素为止，不括最后一个元素。（即，去掉数组最后一位）
    # .view(1, -1)方法将张量重新塑造为一个行数为1、列数自动推断的一个行向量。
    prefix_tokens = prefix_tokens[:-1].view(1, -1)

    # 将目标单词编码
    complex_tokens = model.encode(complex_word)
    # 1.make some change to the original sentence
    # =prefix.strip()+" "+process_string(complex_word.strip()+" "+stored_suffix1.strip())
    # sentence=new_sentence

    # 将原句编码
    sentence_tokens = model.encode(sentence)

    # 将后缀编码
    suffix_tokens = model.encode(suffix1)[:-1]
    # torch.tensor()是一个函数，用于根据给定的数据创建新的张量。
    # 它接受一个数据（如 Python 列表、NumPy 数组等）作为输入，并返回一个新的张量对象。
    suffix_tokens = torch.tensor(suffix_tokens)
    # 将矩阵转换成列表
    suffix_tokens = suffix_tokens.tolist()

    attn_len = len(prefix_tokens[0]) + len(complex_tokens) - 1

    if len((model.tgt_dict.string(prefix_tokens).strip().replace("@@ ", "")).strip().split()) != len(
            prefix.strip().split()):
        print("finding prefix not good before replace mask token!!!")

    outputs, combined_sss, prev_masks, prev_masks2, scores_with_suffix, scores_with_suffix_masks, scores_with_dynamic = model.generate2(
        sentence_tokens.cuda(),  # 句子的tokens
        beam=beam,
        prefix_tokens=prefix_tokens.cuda(),  # 句子前缀的tokens
        attn_len=attn_len,
        # tgt_token=complex_tokens[:-1].tolist(),
        tgt_token=-1,
        suffix_ids=suffix_tokens,  # 句子后缀的tokens
        max_aheads=5
    )
    outputs = outputs.cpu()

    for i in range(len(combined_sss)):
        if combined_sss[i] != []:
            if type(combined_sss[i]) == list:
                combined_sss[i][0] = combined_sss[i][0].to("cpu")
                combined_sss[i][1] = combined_sss[i][1].to("cpu")
            else:
                combined_sss[i] = combined_sss[i].to("cpu")
    try:
        prev_masks = prev_masks.cpu()
        prev_masks2 = prev_masks2.cpu()
    except:
        pass

    scores_with_suffix = scores_with_suffix.cpu()
    scores_with_suffix_masks = scores_with_suffix_masks.cpu()

    embed_scores = give_embedding_scores(
        outputs,
        model.models[0].state_dict()["decoder.embed_tokens.weight"].cpu(),
        complex_tokens=complex_tokens,
        temperature=0.2
    )
    # embed_scores=give_embedding_scores_v2(outputs,model.models[0].state_dict()["decoder.embed_tokens.weight"].cpu(),complex_tokens=complex_tokens,temperature=0.2)
    assert embed_scores.size() == scores_with_suffix[:, :(outputs.size()[-1] - 1)].size()

    outputs, outputs_scores, candis = give_real_scores_ahead(
        model.tgt_dict,
        outputs,
        scores_with_suffix,
        scores_with_suffix_masks,
        suffix_tokens,
        prefix_len=len(prefix_tokens[0]),
        prefix_str=prefix,
        max_ahead=5,
        flag=1
    )

    new_outputs_scores = torch.tensor(outputs_scores)
    outputs_scores = new_outputs_scores
    new_indices = torch.topk(outputs_scores, k=len(outputs_scores), dim=0)[1]

    outputs = [outputs[index1] for index1 in new_indices]
    outputs_scores = [outputs_scores[index1].tolist() for index1 in new_indices]

    output_sentences = [model.decode(x) for x in outputs]
    if output_sentences == []:
        print("find a missing prefix sentence!!!")
        return [], [], [], []

    bertscore_substitutes, ranking_bertscore_substitutes, real_prev_scores = extract_substitute(
        output_sentences,
        sentence, complex_word,
        threshold,
        outputs_scores,
        word_index,
        sentence_words,
        target_pos,
        target_lemma
    )

    bertscore_substitutes = bertscore_substitutes[:50]
    ranking_bertscore_substitutes = ranking_bertscore_substitutes[:50]
    real_prev_scores = real_prev_scores[:50]

    glove_scores = (cal_bart_score(sentence, complex_word, word_index, bertscore_substitutes) +
                    cal_bleurt_score(sentence, complex_word, word_index, bertscore_substitutes))

    return bertscore_substitutes, ranking_bertscore_substitutes, real_prev_scores, glove_scores.tolist()


def lexical_substitute_bart_model(
        model, sentence, sentence_words,
        prefix, word_index, complex_word, target_pos, target_lemma, beam, threshold
):
    index_complex = word_index
    ori_words = sentence_words
    prefix = prefix
    suffix1 = ""
    if index_complex != -1:
        prefix = prefix
        if len(ori_words) > index_complex + 1:
            suffix1 = " ".join(ori_words[index_complex + 1:]).strip()
            suffix1 = suffix1.replace("''", "\"").strip()
            suffix1 = suffix1.replace("``", "\"").strip()
            suffix1 = process_string(suffix1)
            # stored_suffix1=suffix1

            if suffix1.endswith("\""):
                suffix1 = suffix1[:-1]
                suffix1 = suffix1.strip()
            if suffix1.endswith("'"):
                suffix1 = suffix1[:-1]
                suffix1 = suffix1.strip()

        if index_complex + 1 < len(ori_words):
            if ori_words[index_complex + 1] == "-":
                if suffix1.startswith("-"):
                    suffix1 = suffix1[1:].strip()
            if ori_words[index_complex + 1] == "--":
                if suffix1.startswith("--"):
                    suffix1 = suffix1[2:].strip()

            suffix1 = " ".join(suffix1.split(" ")[:2])


        else:
            pass
        # print(prefix)
    else:
        print("*************cannot find the complex word")
        # print(sentence)
        # print(complex_word)
        sentence = sentence.lower()

        return lexical_substitute_bart_model(model, sentence, complex_word, beam, threshold)

    if ori_words[0] == "\"" and index_complex == 1:
        prefix = " " + prefix
    prefix_tokens = model.encode(prefix)
    prefix_tokens = prefix_tokens[:-1].view(1, -1)

    if index_complex != 0:
        complex_tokens = model.encode(" " + complex_word.strip())[1:]
    else:
        complex_tokens = model.encode(" " + complex_word.strip())[1:]
    if index_complex + 1 < len(ori_words):
        if ori_words[index_complex + 1] == "-":
            print("finding a ---------")
            sentence = " ".join(ori_words[:index_complex + 1] + ori_words[index_complex + 2:])

        if ori_words[index_complex + 1] == "--":
            print("finding a ---------")
            sentence = " ".join(ori_words[:index_complex + 1] + ori_words[index_complex + 2:])

    if index_complex == 0:
        sentence_tokens = model.encode(" " + sentence)
    elif index_complex == 1 and ori_words[0] == "\"":
        sentence_tokens = model.encode(" " + sentence)
    else:
        sentence_tokens = model.encode(sentence)

    if suffix1 != "":
        suffix_tokens = model.encode(" " + suffix1.strip())[1:-1]
    else:
        suffix_tokens = model.encode(suffix1.strip())[1:-1]
    # suffix_tokens=torch.tensor(suffix_tokens)
    suffix_tokens = suffix_tokens.tolist()
    attn_len = len(prefix_tokens[0]) + len(complex_tokens) - 1

    outputs, combined_sss, prev_masks, prev_masks2, scores_with_suffix, scores_with_suffix_masks, scores_with_dynamic = model.generate3(
        [sentence_tokens.cuda()],
        beam=beam,
        prefix_tokens=prefix_tokens.cuda(),
        attn_len=attn_len,
        # tgt_token=complex_tokens[:-1].tolist(),
        tgt_token=-1,
        suffix_ids=suffix_tokens,
        max_aheads=5
    )
    outputs = outputs[0].cpu()

    scores_with_suffix = scores_with_suffix.cpu()
    scores_with_suffix_masks = scores_with_suffix_masks.cpu()

    outputs, outputs_scores, candis = give_real_scores_ahead_bart(
        model,
        outputs,
        scores_with_suffix,
        scores_with_suffix_masks,
        suffix_tokens,
        prefix_len=len(prefix_tokens[0]),
        prefix_str=prefix,
        max_ahead=5,
        flag=1
    )

    new_outputs_scores = torch.tensor(outputs_scores)
    outputs_scores = new_outputs_scores
    new_indices = torch.topk(outputs_scores, k=len(outputs_scores), dim=0)[1]
    outputs = [outputs[index1] for index1 in new_indices]

    outputs_scores = [outputs_scores[index1].tolist() for index1 in new_indices]

    output_sentences = [model.bpe.decode(model.task.source_dictionary.string(x)) for x in outputs]
    if output_sentences == []:
        print("find a missing prefix sentence!!!")
        return [], [], [], []

    bertscore_substitutes, ranking_bertscore_substitutes, real_prev_scores = extract_substitute(
        output_sentences,
        sentence, complex_word,
        threshold,
        outputs_scores,
        word_index,
        sentence_words,
        target_pos,
        target_lemma
    )
    # print(pre_scores)

    # for sen in output_sentences:
    #    print(sen)

    bertscore_substitutes = bertscore_substitutes[:50]
    ranking_bertscore_substitutes = ranking_bertscore_substitutes[:50]
    real_prev_scores = real_prev_scores[:50]

    # if index_complex!=0:
    tmp_bertscore_substitutes = [" " + word1.strip() for word1 in bertscore_substitutes]

    glove_scores = (cal_bart_score(sentence, complex_word, word_index, bertscore_substitutes) +
                    cal_bleurt_score(sentence, complex_word, word_index, bertscore_substitutes))

    return bertscore_substitutes, ranking_bertscore_substitutes, real_prev_scores, glove_scores.tolist()


def write_all_results(main_word, instance, target_pos, output_results, substitutes, substitutes_scores,
                      evaluation_metric):
    proposed_words = {}
    for substitute_str, score in zip(substitutes, substitutes_scores):
        substitute_lemma = lemma_word(
            substitute_str,
            target_pos
        ).lower().strip()
        max_score = proposed_words.get(substitute_lemma)
        if max_score is None or score > max_score:
            # if pos_filter(pos_vocab,target_pos,substitute_str,substitute_lemma):
            proposed_words[substitute_lemma] = score

    evaluation_metric.write_results(
        output_results + "_probabilites.txt",
        main_word, instance,
        proposed_words
    )
    evaluation_metric.write_results_p1(
        output_results + "_p1.txt",
        main_word, instance,
        proposed_words
    )
    evaluation_metric.write_results_p1(
        output_results + "_p3.txt",
        main_word, instance,
        proposed_words, limit=3
    )
    evaluation_metric.write_results_lex_oot(
        output_results + ".oot",
        main_word, instance,
        proposed_words, limit=10
    )
    evaluation_metric.write_results_lex_best(
        output_results + ".best",
        main_word, instance,
        proposed_words, limit=1
    )


def main_transformer_model():
    args = {
        'eval_dir': 'data/LS07/ciwater_test_wtgbb/ciwater.preprocessed',
        # 'eval_gold_dir': 'data/LS07/ciwater_test_wtgbb/lst_test.gold',
        'paraphraser_path': 'checkpoints/para/transformer/',
        'paraphraser_model': 'checkpoint_best.pt',
        'bpe': 'subword_nmt',
        'bpe_codes': 'checkpoints/para/transformer/codes.40000.bpe.en',
        'beam': 100,
        'bertscore': -100,
        'output_SR_file': 'ls07_transformer_model_out',
        'output_score_file': 'ls07_transformer_model_scores'
    }

    reader = Reader_lexical()
    # 读取数据
    # 参数
    # --eval_dir $TEST_FILE  (其中TEST_FILE=data/LS07/ciwater_test_wtgbb/lst_test.preprocessed)
    reader.create_feature(args['eval_dir'])
    evaluation_metric = evaluation()
    # 参数
    # --bpe "subword_nmt"
    # --bpe_codes "checkpoints/para/transformer/codes.40000.bpe.en"
    en2en = TransformerModel.from_pretrained(
        args['paraphraser_path'],
        checkpoint_file=args['paraphraser_model'],
        bpe=args['bpe'], bpe_codes=args['bpe_codes']
    ).cuda().eval()

    bert_substitutes_all = []
    real_prev_scores_all = []
    real_embed_scores_all = []
    count_gen = -1

    from tqdm import tqdm

    for main_word in tqdm(reader.words_candidate):
        # main_word = 'side.n'
        count_gen += 1
        for instance in reader.words_candidate[main_word]:
            # 一条instance：'303': [[...]]
            for context in reader.words_candidate[main_word][instance]:
                text = context[1]
                word_index = int(context[2])

                original_text = text
                original_words = text.split(' ')

                # 目标单词
                target_word = text.split(' ')[word_index]

                # 句子前缀。【.strip()方法：去除首尾空格】
                prefix = " ".join(original_words[0:word_index]).strip()

                # 目标单词的词性
                target_pos = main_word.split('.')[-1]

                # 句子后缀
                # suffix = " ".join(original_words[index_word + 1:]).strip()

                # 目标单词的词形还原
                target_lemma = lemma_word(target_word, target_pos=target_pos).lower().strip()

                bert_substitutes, bert_rank_substitutes, real_prev_scores, real_embed_scores = (
                    lexical_substitute_transformer_model(
                        en2en,  # 模型
                        original_text,  # 原句
                        original_words,  # 原句分词后的单词列表
                        prefix,  # 句子前缀
                        word_index,  # 目标单词索引
                        target_word,  # 目标单词
                        target_pos,  # 目标单词词性
                        target_lemma,  # 词形还原后的目标单词
                        args['beam'],
                        args['bertscore']
                    )
                )

                bert_substitutes_all.append(bert_substitutes)
                real_prev_scores_all.append(real_prev_scores)
                real_embed_scores_all.append(real_embed_scores)

    import copy

    range1 = np.arange(0.02, 0.04, 0.02)
    range2_log_softmax = np.arange(1, 2, 1)

    for log_quto in range2_log_softmax:
        work_dir = "ls07_search_results/log." + str(round(log_quto, 2)) + "/"
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        for embed_quto in range1:

            count_gen2 = -1
            count_1 = 0

            tmp_bert_substitutes_all = copy.deepcopy(bert_substitutes_all)
            tmp_real_prev_scores_all = copy.deepcopy(real_prev_scores_all)
            tmp_real_embed_scores_all = copy.deepcopy(real_embed_scores_all)

            for main_word in tqdm(reader.words_candidate):
                count_gen2 += 1

                for instance in reader.words_candidate[main_word]:
                    for context in reader.words_candidate[main_word][instance]:
                        text = context[1]
                        original_text = text
                        original_words = text.split(' ')
                        word_index = int(context[2])
                        target_word = text.split(' ')[word_index]

                        prefix = " ".join(original_words[0:word_index]).strip()
                        target_pos = main_word.split('.')[-1]
                        # suffix = " ".join(original_words[index_word + 1:]).strip()

                        target_lemma = lemma_word(target_word, target_pos=target_pos).lower().strip()

                        tmp_log_embed_scores = torch.tensor(tmp_real_embed_scores_all[count_1])
                        tmp_log_embed_scores = tmp_log_embed_scores.tolist()

                        for k1 in range(len(tmp_real_prev_scores_all[count_1])):
                            tmp_real_prev_scores_all[count_1][k1] = embed_quto * tmp_real_prev_scores_all[count_1][k1] + \
                                                                    tmp_log_embed_scores[k1]
                            pass

                        write_all_results(
                            main_word,
                            instance,
                            target_pos,
                            work_dir + args['output_SR_file'] + ".embed." + str(embed_quto),
                            tmp_bert_substitutes_all[count_1],
                            tmp_real_prev_scores_all[count_1],
                            evaluation_metric
                        )

                        # print("after_score",real_prev_scores_all[count_1][:10])

                        count_1 += 1
            print("*" * 100)

            test_golden_file = "data/LS07/ciwater_test_wtgbb/lst_test.gold"

            output_results = work_dir + args['output_SR_file'] + ".embed." + str(embed_quto)
            results_file = work_dir + args['output_score_file'] + ".embed." + str(embed_quto)

            evaluation_metric.calculation_perl(test_golden_file,
                                               output_results + ".best", output_results + ".oot",
                                               results_file + ".best", results_file + ".oot")
            evaluation_metric.calculation_p1(test_golden_file, output_results + "_p1.txt", results_file + "_p1.txt")
            evaluation_metric.calculation_p3(test_golden_file, output_results + "_p3.txt", results_file + "_p3.txt")


def main_bart_model():
    args = {
        'eval_dir': 'data/LS07/ciwater_test_wtgbb/lst_test.preprocessed',
        'eval_gold_dir': 'data/LS07/ciwater_test_wtgbb/lst_test.gold',
        'paraphraser_path': 'checkpoints/para/bart/',
        'paraphraser_model': 'checkpoint_2_105000.pt',
        'bpe': 'subword_nmt',
        'bpe_codes': 'checkpoints/para/transformer/codes.40000.bpe.en',
        'beam': 100,
        'bertscore': -100,
        'output_SR_file': 'ls07_bart_model_out',
        'output_score_file': 'ls07_bart_model_scores'
    }

    # output_sr_file = open(args.output_SR_file, "w+")
    reader = Reader_lexical()
    reader.create_feature(args['eval_dir'])
    evaluation_metric = evaluation()

    en2en = BARTModel.from_pretrained(args['paraphraser_path'], checkpoint_file=args['paraphraser_model']).cuda().eval()

    bert_substitutes_all = []
    real_prev_scores_all = []
    real_embed_scores_all = []
    count_gen = -1

    from tqdm import tqdm
    for main_word in tqdm(reader.words_candidate):
        count_gen += 1
        for instance in reader.words_candidate[main_word]:
            for context in reader.words_candidate[main_word][instance]:
                text = context[1]
                original_text = text
                original_words = text.split(' ')
                index_word = int(context[2])
                target_word = text.split(' ')[index_word]

                prefix = " ".join(original_words[0:index_word]).strip()
                target_pos = main_word.split('.')[-1]
                # suffix = " ".join(original_words[index_word + 1:]).strip()

                target_lemma = lemma_word(
                    target_word,
                    target_pos=target_pos
                ).lower().strip()

                first_target_word = target_word

                if (index_word > 0
                        and len(original_words[index_word - 1]) != 1
                        and original_words[index_word - 1].endswith(".")
                ):
                    if original_words[index_word - 1] == "u.s.":
                        print("change the u.s.!!!")
                        text = " ".join((original_words[:index_word - 1] + original_words[index_word:])).strip()
                        original_text = text
                        original_words = text.split()
                        index_word = index_word - 1
                        target_word = original_words[index_word]
                        # target_word=text.split()[index_word]
                        prefix = " ".join(original_words[0:index_word]).strip()
                        target_pos = main_word.split('.')[-1]
                        target_lemma = lemma_word(
                            target_word,
                            target_pos=target_pos
                        ).lower().strip()
                        if first_target_word != target_word:
                            print("finding an error in the main fucntion!!!")
                            import pdb
                            pdb.set_trace()

                    elif index_word > 2 and original_words[index_word - 1] == "n." and original_words[
                        index_word - 2] == "u.":
                        print("changeing the u. n.!!!")
                        text = " ".join((original_words[:index_word - 2] + original_words[index_word:])).strip()
                        original_text = text
                        original_words = text.split()
                        index_word = index_word - 2
                        target_word = original_words[index_word]
                        # target_word=text.split()[index_word]
                        target_pos = main_word.split('.')[-1]
                        target_lemma = lemma_word(
                            target_word,
                            target_pos=target_pos
                        ).lower().strip()
                        if first_target_word != target_word:
                            print("finding an error in the main fucntion!!!")
                            import pdb
                            pdb.set_trace()
                    else:
                        original_words[index_word - 1] = original_words[index_word - 1][:-1]
                        text = " ".join(original_words)
                        original_text = text
                        original_words = text.split(' ')
                        index_word = index_word
                        # target_word = text.split(' ')[index_word]
                        target_word = original_words[index_word]

                        prefix = " ".join(original_words[0:index_word]).strip()
                        target_pos = main_word.split('.')[-1]
                        # suffix = " ".join(original_words[index_word + 1:]).strip()

                        target_lemma = lemma_word(
                            target_word,
                            target_pos=target_pos
                        ).lower().strip()

                        if first_target_word != target_word:
                            print("finding an error in the main fucntion!!!")
                            import pdb
                            pdb.set_trace()

                if index_word > 0 and (original_words[index_word - 1] == "t" or original_words[index_word - 1] == "-"):
                    print("change the --")
                    text = " ".join((original_words[:index_word - 1] + original_words[index_word:])).strip()
                    original_text = text
                    original_words = text.split()
                    index_word = index_word - 1
                    # target_word=original_words[insdex_word]
                    target_word = original_words[index_word]
                    prefix = " ".join(original_words[0:index_word]).strip()
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()
                    if first_target_word != target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace()
                    # hadn n't
                if index_word > 1 and (
                        original_words[index_word - 1] == "'t" and original_words[index_word - 2] == "don"):
                    print("change the don 't!!!")
                    # text=" ".join((original_words[:index_word-2]+original_words[index_word:])).strip()
                    original_words[index_word - 1] = "not"
                    original_words[index_word - 2] = "do"
                    text == " ".join(original_words)

                    original_text = text
                    original_words = text.split()
                    target_word = original_words[index_word]
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()
                    if first_target_word != target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace()

                if index_word > 1 and (
                        original_words[index_word - 1] == "n't" and (original_words[index_word - 2] == "couldn")):
                    print("change the could 't!!!")
                    # text=" ".join((original_words[:index_word-2]+original_words[index_word:])).strip()
                    original_words[index_word - 1] = "not"
                    original_words[index_word - 2] = "could"
                    text == " ".join(original_words)

                    original_text = text
                    original_words = text.split()
                    target_word = original_words[index_word]
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()
                    if first_target_word != target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace()

                if index_word > 1 and (
                        original_words[index_word - 1] == "n't" and original_words[index_word - 2] == "don"):
                    print("change the don n't!!!")
                    # text=" ".join((original_words[:index_word-2]+original_words[index_word:])).strip()
                    original_words[index_word - 1] = "not"
                    original_words[index_word - 2] = "do"
                    text == " ".join(original_words)

                    original_text = text
                    original_words = text.split()
                    target_word = original_words[index_word]
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()
                    if first_target_word != target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace()

                if index_word + 2 < len(original_words) and (
                        original_words[index_word + 1] == "hadn" and original_words[index_word + 2] == "n't"):
                    print("change the hadn 't")
                    # text=" ".join((original_words[:index_word-2]+original_words[index_word:])).strip()
                    original_words[index_word + 1] = "had"
                    original_words[index_word + 2] = "not"
                    text == " ".join(original_words)

                    original_text = text
                    original_words = text.split()
                    target_word = original_words[index_word]
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()
                    if first_target_word != target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace()

                bert_substitutes, bert_rank_substitutes, real_prev_scores, real_embed_scores = lexical_substitute_bart_model(
                    en2en,
                    original_text,
                    original_words,
                    prefix,
                    index_word,
                    target_word,
                    target_pos,
                    target_lemma,
                    args['beam'],
                    args['bertscore']
                )

                bert_substitutes_all.append(bert_substitutes)
                real_prev_scores_all.append(real_prev_scores)
                real_embed_scores_all.append(real_embed_scores)

    import numpy as np
    import copy
    import os

    range1 = np.arange(0.02, 0.04, 0.02)
    range2_log_softmax = np.arange(1, 2, 1)
    for log_quto in range2_log_softmax:
        work_dir = "ls07_search_results.bart/log." + str(round(log_quto, 2)) + "/"
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        for embed_quto in range1:
            count_gen2 = -1
            count_1 = 0
            tmp_bert_substitutes_all = copy.deepcopy(bert_substitutes_all)
            tmp_real_prev_scores_all = copy.deepcopy(real_prev_scores_all)
            tmp_real_embed_scores_all = copy.deepcopy(real_embed_scores_all)

            for main_word in tqdm(reader.words_candidate):
                count_gen2 += 1

                for instance in reader.words_candidate[main_word]:
                    for context in reader.words_candidate[main_word][instance]:
                        text = context[1]
                        original_text = text
                        original_words = text.split(' ')
                        index_word = int(context[2])
                        target_word = text.split(' ')[index_word]

                        prefix = " ".join(original_words[0:index_word]).strip()
                        target_pos = main_word.split('.')[-1]
                        # suffix = " ".join(original_words[index_word + 1:]).strip()

                        target_lemma = lemma_word(
                            target_word,
                            target_pos=target_pos
                        ).lower().strip()

                        tmp_log_embed_scores = torch.tensor(tmp_real_embed_scores_all[count_1])

                        tmp_log_embed_scores = tmp_log_embed_scores.tolist()

                        for k1 in range(len(tmp_real_prev_scores_all[count_1])):
                            tmp_real_prev_scores_all[count_1][k1] = embed_quto * tmp_real_prev_scores_all[count_1][k1] + \
                                                                    tmp_log_embed_scores[k1]
                            pass

                        write_all_results(
                            main_word,
                            instance,
                            target_pos,
                            work_dir + args['output_SR_file'] + ".embed." + str(embed_quto),
                            tmp_bert_substitutes_all[count_1],
                            tmp_real_prev_scores_all[count_1],
                            evaluation_metric
                        )

                        count_1 += 1

            test_golden_file = "data/LS07/ciwater_test_wtgbb/lst_test.gold"
            output_results = work_dir + args['output_SR_file'] + ".embed." + str(embed_quto)
            results_file = work_dir + args['output_score_file'] + ".embed." + str(embed_quto)
            evaluation_metric.calculation_perl(
                test_golden_file,
                output_results + ".best",
                output_results + ".oot",
                results_file + ".best",
                results_file + ".oot"
            )
            evaluation_metric.calculation_p1(
                test_golden_file,
                output_results + "_p1.txt",
                results_file + "_p1.txt"
            )

            evaluation_metric.calculation_p3(
                test_golden_file,
                output_results + "_p3.txt",
                results_file + "_p3.txt"
            )


if __name__ == "__main__":
    main_transformer_model()
    # main_bart_model()
