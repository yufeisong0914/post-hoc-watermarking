import copy
import spacy
import random
import util_string
from tqdm import tqdm
import torch
from transformers import BertForMaskedLM, BertTokenizer
import os
import string

source_root = "/home/haojifei/develop_tools/transformers/models/"


def generate_unique_random_integers(n, min, max, seed=None, excluded_list=None):
    """
    在[min,max]内生成n个整数，且这n个数字不在已给出的数字列表excluded_list中
    Args:
        n: 个数
        min: 最小
        max: 最大
        seed: 随机种子
        excluded_list: 剔除表
    Returns: n个不重复的整数s
    """
    if n > (max - min + 1):  # 确保要生成的数量不超过范围内的整数总数
        raise ValueError("Cannot generate more unique integers than the range size.")

    if excluded_list is not None:
        all_numbers = list(set(range(min, max)) - set(excluded_list))
    else:
        all_numbers = list(set(range(min, max)))

    random.seed(seed)
    return random.sample(all_numbers, n)


def punctuation_detect(doc_tokens):
    """
    返回句子中所有标点符号的下标
    Args:
        doc_tokens: 经过spaCy处理过的文本的Token list
    Returns: 所有标点符号的下标list
    """
    punct_index = []
    for i, token in enumerate(doc_tokens):
        if token.is_punct:
            punct_index.append(i)
    return punct_index


def number_detect(doc_tokens):
    number_index = []
    for i, token in enumerate(doc_tokens):
        s = token.text
        for c in s:
            if c.isdigit():
                number_index.append(i)
                break
    return number_index


def deletion_attack(start, end, attack_rate_list, doc_text, seed=None):
    doc_tokens = [token for token in doc_text]
    # 去除标点符号
    punct_index = punctuation_detect(doc_tokens)
    # 最大的攻击概率
    attack_rate_max = attack_rate_list[len(attack_rate_list) - 1]
    # 根据最大攻击概率计算将要攻击的单词数
    n = attack_rate_max * (len(doc_tokens) - len(punct_index))
    # 生成被攻击单词下标
    random_int_list = generate_unique_random_integers(int(n), start, end, seed=seed, excluded_list=punct_index)
    # print(random_int_list)

    # deleted_mask = [[1] * len(doc_tokens)] * len(attack_rate_list)
    deleted_mask = []
    for _ in range(len(attack_rate_list)):
        row = [1] * len(doc_tokens)
        deleted_mask.append(row)
    # print(deleted_mask)

    for i, attack_rate in enumerate(attack_rate_list):
        n = attack_rate * (len(doc_tokens) - len(punct_index))
        for index in random_int_list[:int(n)]:
            deleted_mask[i][index] = 0
    # print(deleted_mask)
    corrupted_text_list = []
    for i, attack_rate in enumerate(attack_rate_list):
        corrupted_text = ""
        for j, token in enumerate(doc_tokens):
            if deleted_mask[i][j]:  # 过滤掉mask=0的token，连接mask=1的token
                corrupted_text = corrupted_text + " " + token.text
        corrupted_text = util_string.preprocess_string(corrupted_text)
        corrupted_text_list.append(corrupted_text)
    return corrupted_text_list


def filter_mask_index(spacy_model, doc_tokens, text_tokens, i, good_candidates_index, candidates):
    """
    Args:
        good_candidates_index: 好候选词的下标
        candidates: 候选词列表
        i: 候选词在原句中的位置
    Returns: 经过筛选的好候选词下标
    """
    # print(i)
    target_token = text_tokens[i]
    good_candidates_index_return = []
    for j in good_candidates_index:
        # 筛选合适的候选词，放入原本的句子中
        if (candidates[j].lower() != target_token.lower()
                and not candidates[j].startswith("##")
                and candidates[j] != '[UNK]'
                and candidates[j] not in string.punctuation
        ):
            text_tokens[i] = candidates[j].strip("'")
            temp_sentence = " ".join(text_tokens[:]).strip()
            temp_sentence = util_string.preprocess_string(temp_sentence)
            temp_sentence_doc = spacy_model(temp_sentence)
            # print(temp_sentence_doc.text)
            temp_sentence_doc_tokens = [token for token in temp_sentence_doc]

            # 查看候选词在句子中的依存关系是否与原词一致
            try:
                if doc_tokens[i].dep_ == temp_sentence_doc_tokens[i].dep_:
                    good_candidates_index_return.append(j)
                    break
            except IndexError:
                bobo = 1
                print("------error------ori sentence length:", len(doc_tokens))
                print(bobo)
                print("------error------with sub length:", len(temp_sentence_doc_tokens))

    return good_candidates_index_return


def candidates_gen(device, tokenizer, model, doc_tokens, index_space, top_k=8, dropout_prob=0.3):
    text_tokens = [token.text for token in doc_tokens]

    # tokens_text_to_bert = ['[CLS]'] + text_tokens + ['[SEP]']
    # index_space = [i + 1 for i in index_space]

    input_ids_bert = tokenizer.convert_tokens_to_ids(text_tokens)

    # Create a tensor of input IDs
    input_tensor = torch.tensor([input_ids_bert]).to(device)

    with torch.no_grad():
        embeddings = model.bert.embeddings(input_tensor.repeat(len(index_space), 1))

    # dropout = nn.Dropout2d(p=dropout_prob)
    #
    # masked_indices = torch.tensor(index_space).to(device)
    # embeddings[torch.arange(len(index_space)), masked_indices] = dropout(
    #     embeddings[torch.arange(len(index_space)), masked_indices]
    # )

    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings)

    all_processed_tokens = []
    for i, masked_token_index in enumerate(index_space):
        if input_ids_bert[masked_token_index] in tokenizer.all_special_ids:
            all_processed_tokens.append(text_tokens[masked_token_index])  # 是tokenizer定义的special_ids就用原词
        else:
            predicted_logits = outputs[0][i][masked_token_index]
            # Set the number of top predictions to return
            # Get the top n predicted tokens and their probabilities
            probs = torch.nn.functional.softmax(predicted_logits, dim=-1)
            top_n_probs, top_n_indices = torch.topk(probs, top_k)
            top_n_tokens = tokenizer.convert_ids_to_tokens(top_n_indices.tolist())
            # print(top_n_tokens)

            good_candidates_index = list(range(0, len(top_n_tokens)))
            text_tokens_temp = copy.deepcopy(text_tokens)
            good_candidates_index = filter_mask_index(
                nlp, doc_tokens, text_tokens_temp, masked_token_index, good_candidates_index, top_n_tokens
            )  # masked_token_index - 1

            if good_candidates_index:
                all_processed_tokens.append(top_n_tokens[good_candidates_index[0]])
            else:
                all_processed_tokens.append(text_tokens[masked_token_index])  # 替换结果都不合适就用原词

    return all_processed_tokens


def substitution_attack(device, tokenizer, model, start, end, attack_rate_list, doc_text, seed=None):
    doc_tokens = [token for token in doc_text]
    # 去除标点符号
    punct_index = punctuation_detect(doc_tokens)
    # 去除数字
    number_index = number_detect(doc_tokens)
    # 最大的攻击概率
    attack_rate_max = attack_rate_list[len(attack_rate_list) - 1]
    # 根据最大攻击概率计算将要攻击的单词数
    n = int(attack_rate_max * (len(doc_tokens) - len(punct_index) - len(number_index)))
    # print(n)
    # 生成被攻击单词下标
    random_int_list = generate_unique_random_integers(
        n, start, end, seed=seed, excluded_list=punct_index + number_index
    )
    # print(random_int_list)
    # 使用bert生成替换词
    all_substitutes = candidates_gen(device, tokenizer, model, doc_tokens, random_int_list, top_k=32, dropout_prob=0.3)
    # print(all_substitutes)

    tokens = [token.text for token in doc_text]
    # 将bert生成的替换词换到句子中
    corrupted_text_list, random_attack_index_list, all_substitutes_list = [], [], []
    for attack_rate in attack_rate_list:
        temp_tokens = copy.deepcopy(tokens)
        n = int(attack_rate * (len(temp_tokens) - len(punct_index) - len(number_index)))
        for j in range(n):
            temp_tokens[random_int_list[j]] = all_substitutes[j]
            # print(temp_tokens[random_int_list[j]], random_int_list[j], all_substitutes[j])
        corrupted_text = " ".join(temp_tokens).strip()
        corrupted_text = util_string.preprocess_string(corrupted_text)
        corrupted_text_list.append(corrupted_text)
        random_attack_index_list.append(random_int_list[:n])
        all_substitutes_list.append(all_substitutes[:n])

    return corrupted_text_list, random_attack_index_list, all_substitutes_list


def generate_corrupted(
        spacy_model, dataset_name: str, watermark_model_tag: str,
        attack_rate_list: list[float], attack_mode='delete', output_dir='ciwater_output'
):
    """
    生成被攻击过的文本
    Args:
        spacy_model: spaCy模型，用来分词
        dataset_name: 数据集名称
        watermark_model_tag: 水印方法名称
        attack_mode: 攻击方式
        attack_rate_list: 攻击概率列表(已经从小到大排好序的)
        output_dir: soso
    Returns: True/False
    """
    file_dir = '../' + output_dir + '/' + dataset_name + '/'
    file = open(file_dir + watermark_model_tag + '.txt', 'r')  # 读取将要被攻击的文件
    # file = open('./a_file.txt', 'r')
    print("read: ", file_dir + watermark_model_tag + '.txt')

    file_dir = file_dir + 'attack_' + attack_mode
    os.makedirs(file_dir, exist_ok=True)

    file_dir = file_dir + '/' + watermark_model_tag  # todo:2

    file_p, file_i_p, file_s_p = [], [], []
    for attack_rate in attack_rate_list:  # 根据不同的攻击概率生成攻击结果
        temp_file_path = file_dir + '_' + str(attack_rate)  # str(int(attack_rate * 100))

        file_attack = open(temp_file_path + '.txt', 'w')
        print("create: ", temp_file_path + '.txt')
        file_p.append(file_attack)

        if attack_mode == 'substitute':
            file_attack_i = open(temp_file_path + '_index.txt', 'w')
            print("create: ", temp_file_path + '_index.txt')
            file_attack_s = open(temp_file_path + '_' + attack_mode + '.txt', 'w')
            print("create: ", temp_file_path + '_' + attack_mode + '.txt')
            file_i_p.append(file_attack_i)
            file_s_p.append(file_attack_s)

    bar = tqdm(total=200)
    line = file.readline()
    if attack_mode == 'substitute':
        print("Substituting ...")
        while line:
            # line = util_string.preprocess_string(line)
            # line = util_string.remove_adjacent_commas(line)
            # line = util_string.remove_adjacent_commas(line)
            doc_line = spacy_model(line)
            corrupted_text_list, random_attack_index_list, all_substitutes_list = substitution_attack(
                device, tokenizer, model, 0, len(doc_line), attack_rate_list, doc_line, 18
            )
            for i in range(len(attack_rate_list)):
                file_p[i].write(corrupted_text_list[i] + '\n')
                file_i_p[i].write(str(random_attack_index_list[i]) + '\n')
                file_s_p[i].write(str(all_substitutes_list[i]) + '\n')
            line = file.readline()
            bar.update(1)
    else:  # attack_mode == 'delete':
        print("Deleting ...")
        while line:
            doc_line = spacy_model(line)
            corrupted_text_list = deletion_attack(0, len(doc_line), attack_rate_list, doc_line, 18)
            for i in range(len(attack_rate_list)):
                file_p[i].write(corrupted_text_list[i] + '\n')
            line = file.readline()
            bar.update(1)
    print("Done!")

    file.close()
    for i in range(len(attack_rate_list)):
        file_p[i].close()
        if attack_mode == 'substitute':
            file_i_p[i].close()
            file_s_p[i].close()
    return True


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')

    # rate_list = [0.025, 0.05, 0.1, 0.2, 0.4]  # todo: 需要吗？0.3, 0.5

    rate_list = [0.025, 0.05, 0.1, 0.2, 0.4, 0.3, 0.5]
    rate_list.sort(key=float)

    source_list = ["wiki_csai", "open_qa", "reddit_eli5", "medicine"]  # todo:
    tag_list = ["808083", "758083", "858083"]  # todo: "ciwater5010", "ciwater5015", "ciwater5020", "808083", "758083", "858083"
    attack_list = ["delete", "substitute"]  # todo: "delete", "substitute"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(source_root + 'google-bert/bert-base-cased')
    model = BertForMaskedLM.from_pretrained(
        source_root + 'google-bert/bert-base-cased', output_hidden_states=True
    ).to(device)

    for source in source_list:
        for tag in tag_list:
            for attack in attack_list:
                generate_corrupted(nlp, source, tag, rate_list, attack_mode=attack)
