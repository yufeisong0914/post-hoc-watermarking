import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from fairseq.models.transformer import TransformerModel
import ciwater_util_string


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


def lexical_substitute_transformer_model(
        model, ori_sen,
        prefix, mask_word, suffix,
        beam, threshold
):
    # 将原句编码
    sentence_tokens = model.encode(ori_sen)  # tensor([..., 2])

    # 将前缀编码
    prefix_tokens = model.encode(prefix)  # tensor([..., 2])
    # [:-1]表示从序列的第一个元素开始，倒数第二个元素为止，不括最后一个元素。（即，去掉数组最后一位，因为encode()后最后以为是结束位）
    # .view(1, -1)方法将张量重新塑造为一个行数为1、列数自动推断的一个行向量。
    prefix_tokens = prefix_tokens[:-1].view(1, -1)  # tensor([[...]])

    # 将目标单词编码
    mask_word_tokens = model.encode(mask_word)  # tensor([..., 2])

    # 将后缀编码
    suffix_tokens = model.encode(suffix)[:-1]  # tensor([...])
    # torch.tensor()是一个函数，用于根据给定的数据创建新的张量。
    # 它接受一个数据（如 Python 列表、NumPy 数组等）作为输入，并返回一个新的张量对象。
    # suffix_tokens = torch.tensor(suffix_tokens)
    # 将矩阵转换成列表
    suffix_tokens = suffix_tokens.tolist()

    length_prefix_and_mask = len(prefix_tokens[0]) + len(mask_word_tokens) - 1

    if len((model.tgt_dict.string(prefix_tokens).strip().replace("@@ ", "")).strip().split()) != len(
            prefix.strip().split()
    ):
        print("finding prefix not good before replace mask token!!!")

    outputs, combined_sss, prev_masks, prev_masks2, scores_with_suffix, scores_with_suffix_masks, scores_with_dynamic = model.generate2(
        sentence_tokens.cuda(),  # 句子的tokens
        beam=beam,
        prefix_tokens=prefix_tokens.cuda(),  # 句子前缀的tokens
        attn_len=length_prefix_and_mask,
        # tgt_token=complex_tokens[:-1].tolist(),
        tgt_token=-1,
        suffix_ids=suffix_tokens,  # 句子后缀的tokens
        max_aheads=5
    )
    outputs = outputs.cpu()

    scores_with_suffix = scores_with_suffix.cpu()
    scores_with_suffix_masks = scores_with_suffix_masks.cpu()

    # 排序后的结构，但是包含目标词的其他形态
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

    return outputs, candis

    # new_outputs_scores = torch.tensor(outputs_scores)
    # outputs_scores = new_outputs_scores
    # new_indices = torch.topk(outputs_scores, k=len(outputs_scores), dim=0)[1]
    #
    # outputs = [outputs[index1] for index1 in new_indices]
    # outputs_scores = [outputs_scores[index1].tolist() for index1 in new_indices]
    #
    # output_sentences = [model.decode(x) for x in outputs]
    # # 如果output_sentences为空
    # if not output_sentences:
    #     print("find a missing prefix sentence!!!")
    #     return [], [], [], []
    #
    # # 剔除了目标词的其他词形
    # bertscore_substitutes, ranking_bertscore_substitutes, real_prev_scores = extract_substitute(
    #     output_sentences,
    #     ori_sen,
    #     mask_word,
    #     threshold,
    #     outputs_scores,
    #     mask_word_index,
    #     ori_sen_words,
    #     mask_word_pos,
    #     mask_word_lemma
    # )
    #
    # bertscore_substitutes = bertscore_substitutes[:50]
    # ranking_bertscore_substitutes = ranking_bertscore_substitutes[:50]
    # real_prev_scores = real_prev_scores[:50]
    #
    # glove_scores = (cal_bart_score(ori_sen, mask_word, mask_word_index, bertscore_substitutes) +
    #                 cal_bleurt_score(ori_sen, mask_word, mask_word_index, bertscore_substitutes))
    #
    # return bertscore_substitutes, ranking_bertscore_substitutes, real_prev_scores, glove_scores.tolist()


if __name__ == "__main__":
    args = {
        'eval_dir': 'data/ciwater/ciwater_test_wtgbb/lst_test.preprocessed',
        'paraphraser_path': 'checkpoints/para/transformer/',
        'paraphraser_model': 'checkpoint_best.pt',
        'bpe': 'subword_nmt',
        'bpe_codes': 'checkpoints/para/transformer/codes.40000.bpe.en',
        'beam': 100,
        'bertscore': -100,
    }

    model = TransformerModel.from_pretrained(
        args['paraphraser_path'],
        checkpoint_file=args['paraphraser_model'],
        bpe=args['bpe'],
        bpe_codes=args['bpe_codes']
    ).cuda().eval()

    ori_sen = "When updating to a newer version of spaCy, it’s generally recommended to start with a clean virtual environment."

    # ori_sen_words = ori_sen.split(' ') # 使用.split(' ')将原句根据空格分为单词数组
    ori_sen_words = nltk.word_tokenize(ori_sen)  # 使用nltk进行分词

    # 目标单词在句子中的下标
    target_word_index = 4
    # 目标单词
    target_word = ori_sen_words[target_word_index]

    # 句子前缀【.strip()方法：去除首尾空格】
    prefix = " ".join(ori_sen_words[0:target_word_index]).strip()

    # 目标单词的词性
    # target_pos = main_word.split('.')[-1]
    target_pos = pos_tag(ori_sen_words, tagset='universal')[target_word_index][1]
    pos_coco = {'ADJ': 'a', 'ADJ-SAT': 's', 'ADV': 'r', 'NOUN': 'n', 'VERB': 'v'}
    target_pos = pos_coco[target_pos]

    # 句子后缀
    suffix = ""
    if target_word_index > 0:
        # 保证目标单词的后面有2个token作为后缀
        if len(ori_sen_words) > target_word_index + 1:
            # .strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            suffix = " ".join(ori_sen_words[target_word_index + 1:]).strip()  # 去除首尾空格
            suffix = suffix.replace("\'\'", "\"").strip()
            suffix = suffix.replace("``", "\"").strip()
            suffix = ciwater_util_string.preprocess_string(suffix)

            # 去除句子后面的引号或单引号
            if suffix.endswith("\"") or suffix.endswith("'"):
                suffix = suffix[:-1]
                suffix = suffix.strip()

            # 仅选择目标词后的2个单词作为后缀
            suffix = " ".join(suffix.split(" ")[:2])
        else:
            pass
    else:
        print("-" * 32, "can not find the target word!")

    # 目标单词的词形还原
    # target_lemma = lemmatize_word(target_word, target_pos=target_pos).lower().strip()
    # nltk.wordnet.WordNetLemmatizer()方法是NLTK库中的一个词形还原器类，用于将单词转换为它们的基本词形。
    wordnet_lemmatizer = WordNetLemmatizer()
    target_lemma = wordnet_lemmatizer.lemmatize(target_word, pos=target_pos).lower().strip()

    # bert_substitutes, bert_rank_substitutes, real_prev_scores, real_embed_scores = (
    #     lexical_substitute_transformer_model(
    #         model,  # 模型
    #         ori_sen,  # 原句
    #         ori_sen_words,  # 原句分词后的单词列表
    #         prefix,  # 句子前缀
    #         target_word_index,  # 目标单词索引
    #         target_word,  # 目标单词
    #         target_pos,  # 目标单词词性 Part-of-Speech
    #         target_lemma,  # 目标单词 词形还原（Lemmatization）
    #         suffix,
    #         args['beam'],
    #         args['bertscore']
    #     )
    # )
    # print(bert_substitutes)
