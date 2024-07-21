import os

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from bart_score import BARTScorer
import bert_score
from bert_score import BERTScorer
from nltk.translate import meteor_score
from nltk.tokenize import sent_tokenize
import spacy
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import f1_score

nlp = spacy.load('en_core_web_sm')
# plt_color = ['red', 'orange', 'green', 'blue', 'gray', 'purple']
# plt_color = ['blue', 'red', 'gray', 'purple']  # todo: , 'green', 'orange' for legacy
# plt_color = ['green', 'orange', 'red', 'blue', 'gray', 'purple']
plt_color = ['lightsteelblue', 'blue', 'lightcoral', 'red']
plt_color_i = 0
plt_line_style = [(0, (1, 1)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1))]
plt_line_mark = ['o', '^', 's', 'P', 'X', 'p', '*']

# 设置全局字体大小
TINY_SIZE = 8
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

# 设置各个部分的字体大小
rcParams['font.size'] = BIGGER_SIZE  # 默认字体大小
rcParams['axes.titlesize'] = BIGGER_SIZE  # 标题字体大小
rcParams['axes.labelsize'] = BIGGER_SIZE  # 坐标轴标签字体大小
rcParams['xtick.labelsize'] = BIGGER_SIZE  # x轴刻度标签字体大小
rcParams['ytick.labelsize'] = BIGGER_SIZE  # y轴刻度标签字体大小
rcParams['legend.fontsize'] = BIGGER_SIZE  # 图例字体大小
rcParams['figure.titlesize'] = BIGGER_SIZE  # 图形标题字体大小


def _calculate_similarity_for_corrupted(
        evaluation_tokenizer, evaluation_model,
        file_original_text, file_watermarked, file_sim
):
    """
    有缺陷，max_length=512 截断，导致过长的第二个句子被截断，使相似度极低
    Args:
        file_original_text: 原文
        file_watermarked: 水印文本(被攻击过)
        file_sim: 记录相似度
    Returns: void
    """
    original_text = file_original_text.readline()
    watermarked_text = file_watermarked.readline()

    all_sentences = []
    while watermarked_text:
        all_sentences.extend([watermarked_text.strip("\n") + '</s></s>' + original_text.strip("\n")])
        original_text = file_original_text.readline()
        watermarked_text = file_watermarked.readline()

    batch_size = 64
    all_relatedness_scores = []  # 计算相似度
    for i in range(0, len(all_sentences), batch_size):  # 步长是batch_size
        encoded_dict = evaluation_tokenizer.batch_encode_plus(
            all_sentences[i: i + batch_size], padding=True, truncation=True, max_length=512, return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids'].to(device)
        attention_masks = encoded_dict['attention_mask'].to(device)

        with torch.no_grad():
            outputs = evaluation_model(input_ids=input_ids, attention_mask=attention_masks)
            logits = outputs[0]
        probs = torch.softmax(logits, dim=1)
        batch_relatedness_scores = probs[:, 2]  # tensor(...)
        temp_list = batch_relatedness_scores.tolist()
        all_relatedness_scores = all_relatedness_scores + temp_list

    for score in all_relatedness_scores:
        file_sim.write(str(score) + '\n')


def _calculate_similarity_for_corrupted_v2(
        evaluation_tokenizer, evaluation_model,
        file_original_text, file_watermarked, file_sim
):
    """
    计算水印文本与原文的文本相似度
    Args:
        file_original_text: 原文
        file_watermarked: 水印文本(被攻击过)
        file_sim: 存放结果
    Returns: None
    """
    original_text = file_original_text.readline()
    watermarked_text = file_watermarked.readline()

    bar = tqdm(total=200)
    jump_flag = False
    while watermarked_text:
        batch_sentences = []
        # 使用NLTK将每一行的文本进行分句，对齐颗粒度
        watermarked_text_sentences = sent_tokenize(watermarked_text)
        length_watermarked_text = len(watermarked_text_sentences)
        original_text_sentences = sent_tokenize(original_text)
        length_original_text = len(original_text_sentences)

        # 如果没有对齐
        if length_watermarked_text != length_original_text:
            # 使用spacy将每一行的文本进行分句，对齐颗粒度
            doc1 = nlp(watermarked_text)
            watermarked_text_sentences = [sentence.text for sentence in doc1.sents]
            length_watermarked_text = len(watermarked_text_sentences)
            doc2 = nlp(original_text)
            original_text_sentences = [sentence.text for sentence in doc2.sents]
            length_original_text = len(original_text_sentences)

            # 如果还是没有对齐
            if length_watermarked_text != length_original_text:
                print("please dui-qi ke-li-du!")
                batch_sentences.extend(
                    [watermarked_text.strip("\n") + '</s></s>' + original_text.strip("\n")]
                )
                jump_flag = True

        if not jump_flag:
            for i in range(length_watermarked_text):
                # 水印句子</s></s>原句
                batch_sentences.extend(
                    [watermarked_text_sentences[i].strip("\n") + '</s></s>' + original_text_sentences[i].strip(
                        "\n")]
                )
        encoded_dict = evaluation_tokenizer.batch_encode_plus(
            batch_sentences, padding=True, truncation=True, max_length=512, return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids'].to(device)
        attention_masks = encoded_dict['attention_mask'].to(device)

        with torch.no_grad():
            outputs = evaluation_model(input_ids=input_ids, attention_mask=attention_masks)
            logits = outputs[0]
        probs = torch.softmax(logits, dim=1)  # tensor:(length_watermarked_text, 3)
        relatedness_scores = probs[:, 2]  # tensor:(length_watermarked_text, 1) = tensor(list[float])
        relatedness_scores_list = relatedness_scores.tolist()
        score_sum = 0
        for score in relatedness_scores_list:
            score_sum += score
        file_sim.write(str(score_sum / length_watermarked_text) + '\n')

        watermarked_text = file_watermarked.readline()
        original_text = file_original_text.readline()
        bar.update(1)
        jump_flag = False


def _calculate_similarity_by_sentran(scorer, file_original_text, file_watermarked, file_sim):
    """
    计算水印文本与原文的文本相似度
    Args:
        scorer: 模型
        file_original_text: 原文
        file_watermarked: 水印文本
        file_sim: 存放结果
    Returns: None
    """

    original = file_original_text.readlines()
    watermarked = file_watermarked.readlines()

    embeddings_ori = scorer.encode(original)
    # print(embeddings_ori.shape)

    embeddings_wmd = scorer.encode(watermarked)
    # print(embeddings_wmd.shape)

    sim_scores = scorer.similarity(embeddings_ori, embeddings_wmd)
    # print(sim_scores)

    s = 0
    for i in range(len(sim_scores)):
        # print(sim_scores[i][i])
        s += sim_scores[i][i]
        file_sim.write(str(float(sim_scores[i][i])) + '\n')

    print(s / len(sim_scores))


def _calculate_similarity_by_bert_score(scorer, file_original_text, file_watermarked, file_sim):
    """
    计算水印文本与原文的文本相似度
    Args:
        file_original_text: 原文
        file_watermarked: 水印文本(被攻击过)
        file_sim: 存放结果
    Returns: None
    """

    original_text = file_original_text.readline()
    watermarked_text = file_watermarked.readline()

    bar = tqdm(total=200)
    jump_flag = False
    while watermarked_text:
        ref, cand = [], []
        # 使用NLTK将每一行的文本进行分句，对齐颗粒度
        watermarked_text_sentences = sent_tokenize(watermarked_text)
        length_watermarked_text = len(watermarked_text_sentences)
        original_text_sentences = sent_tokenize(original_text)
        length_original_text = len(original_text_sentences)

        # 如果没有对齐
        if length_watermarked_text != length_original_text:
            # 使用spacy将每一行的文本进行分句，对齐颗粒度
            doc1 = nlp(watermarked_text)
            watermarked_text_sentences = [sentence.text for sentence in doc1.sents]
            length_watermarked_text = len(watermarked_text_sentences)
            doc2 = nlp(original_text)
            original_text_sentences = [sentence.text for sentence in doc2.sents]
            length_original_text = len(original_text_sentences)

            # 如果还是没有对齐
            if length_watermarked_text != length_original_text:
                print("please dui-qi ke-li-du!")
                ref.extend([original_text.strip("\n")])
                cand.extend([watermarked_text.strip("\n")])
                jump_flag = True

        if not jump_flag:
            for i in range(length_watermarked_text):
                ref.extend([original_text_sentences[i].strip("\n")])
                cand.extend([watermarked_text_sentences[i].strip("\n")])

        P, R, F1 = scorer.score(cand, ref, verbose=True)

        sim_scores_list = F1.tolist()
        score_sum = 0
        for score in sim_scores_list:
            score_sum += score
        file_sim.write(str(score_sum / length_watermarked_text) + '\n')

        watermarked_text = file_watermarked.readline()
        original_text = file_original_text.readline()
        bar.update(1)
        jump_flag = False


def _calculate_similarity_by_bert_score_v2(scorer, file_original_text, file_watermarked, file_sim):
    """
    计算水印文本与原文的文本相似度
    Args:
        scorer: 模型
        file_original_text: 原文
        file_watermarked: 水印文本
        file_sim: 存放结果
    Returns: None
    """

    original = file_original_text.readlines()
    watermarked = file_watermarked.readlines()

    P, R, F1 = scorer.score(watermarked, original, verbose=True)
    # print(sim_scores)

    for i in range(len(F1)):
        file_sim.write(str(float(F1[i])) + '\n')


def _calculate_similarity_by_bart_score(scorer, file_original_text, file_watermarked, file_sim):
    """
    计算水印文本与原文的文本相似度
    Args:
        scorer: 模型
        file_original_text: 原文
        file_watermarked: 水印文本
        file_sim: 存放结果
    Returns: None
    """

    original = file_original_text.readlines()
    watermarked = file_watermarked.readlines()

    sim_scores = scorer.score(watermarked, original)
    # print(sim_scores)

    for i in range(len(sim_scores)):
        file_sim.write(str(sim_scores[i]) + '\n')


def _calculate_similarity(
        evaluation_tokenizer, evaluation_model,
        file_original_text, file_watermarked, file_label, file_sim
):
    """
    计算水印文本与原文的文本相似度
    Args:
        file_original_text: 原文
        file_watermarked: 水印文本
        file_label: 水印标签
        file_sim: 存放结果
    Returns: None
    """
    original_text = file_original_text.readline()
    watermarked_text = file_watermarked.readline()
    text_label = file_label.readline()

    bar = tqdm(total=100)
    while watermarked_text:
        batch_sentences = []

        watermarked_text_sentences = sent_tokenize(watermarked_text)
        length_watermarked_text = len(watermarked_text_sentences)
        original_text_sentences = sent_tokenize(original_text)
        length_original_text = len(original_text_sentences)

        if length_watermarked_text != length_original_text:
            print("please dui-qi ke-li-du!")
            exit(-1)

        if int(text_label.strip('\n')) == 1:
            for i in range(length_watermarked_text):
                # 水印句子</s></s>原句
                batch_sentences.extend(
                    [watermarked_text_sentences[i].strip("\n") + '</s></s>' + original_text_sentences[i].strip("\n")]
                )
            encoded_dict = evaluation_tokenizer.batch_encode_plus(
                batch_sentences, padding=True, truncation=True, max_length=512, return_tensors='pt'
            )
            input_ids = encoded_dict['input_ids'].to(device)
            attention_masks = encoded_dict['attention_mask'].to(device)

            with torch.no_grad():
                outputs = evaluation_model(input_ids=input_ids, attention_mask=attention_masks)
                logits = outputs[0]
            probs = torch.softmax(logits, dim=1)  # tensor:(length_watermarked_text, 3)
            relatedness_scores = probs[:, 2]  # tensor:(length_watermarked_text, 1) = tensor(list[float])
            relatedness_scores_list = relatedness_scores.tolist()
            score_sum = 0
            for score in relatedness_scores_list:
                score_sum += score
            file_sim.write(str(score_sum / length_watermarked_text) + '\n')
        else:
            file_sim.write(str(1) + '\n')

        watermarked_text = file_watermarked.readline()
        text_label = file_label.readline()
        original_text = file_original_text.readline()
        bar.update(1)


def _calculate_meteor(file_original_text, file_watermarked, file_label, file_meteor, attack_mode=None):
    original_text = file_original_text.readline()  # 读取第一行
    watermarked_text = file_watermarked.readline()  # 读取第一行
    text_label = file_label.readline()  # 读取第一行

    bar = tqdm(total=200)
    while watermarked_text:
        if int(text_label.strip('\n')) == 1:
            temp_meteor = meteor_score.meteor_score([original_text.split()], watermarked_text.split())
            file_meteor.write(str(temp_meteor) + '\n')
        else:
            if attack_mode is not None:
                temp_meteor = meteor_score.meteor_score([original_text.split()], watermarked_text.split())
                file_meteor.write(str(temp_meteor) + '\n')
            else:
                file_meteor.write(str(1) + '\n')
        original_text = file_original_text.readline()
        watermarked_text = file_watermarked.readline()
        text_label = file_label.readline()
        bar.update(1)


def calculate_ppl(evaluation_tokenizer, evaluation_model, file, file_ppl):
    bar = tqdm(total=200)
    text = file.readline()
    while text:
        input_ids = evaluation_tokenizer.encode(text, return_tensors='pt').to(device)
        with torch.no_grad():
            output = evaluation_model(input_ids, labels=input_ids)
            log_likelihood = output.loss  # 获取对数似然
            perplexity = torch.exp(log_likelihood)  # 计算困惑度
        file_ppl.write(str(perplexity.item()) + '\n')
        text = file.readline()
        bar.update(1)


def draw_ppl(files_ppl, file_tags, title):
    all_useful_values = []
    for file in files_ppl:
        ppl_values = file.readlines()
        useful_values = []
        flag = 1
        for value in ppl_values:
            if flag % 2 == 1:
                useful_values.append(float(value.strip('\n')))
            flag += 1
        all_useful_values.append(useful_values)
    # 创建箱型图
    plt.boxplot(all_useful_values, showfliers=False)
    plt.ylim([0, 50])
    x_ori = []
    for i in range(len(all_useful_values)):
        x_ori.append(i + 1)
    plt.xticks(x_ori, file_tags)  # [1, 2, ...] ['xxx', 'xxx', ...]
    plt.ylabel('PPL')
    plt.savefig(title + '_ppl.png')
    plt.show()


def calculate_text_quality_score(
        dataset_name: str, watermark_model_tag: str, text_quality_tag='similarity', sim_model_name='similarity',
        attack_mode=None, attack_rate=0.1, output_dir="ciwater_output"
):
    """
    Args:
        sim_model_name:
        dataset_name: 数据集
        watermark_model_tag: 水印方法
        text_quality_tag: 水印质量tag:['ppl', 'similarity', 'meteor']
        attack_mode: 攻击模式
        attack_rate: 攻击率
        output_dir:
    Returns: None
    """
    print('-' * 32)
    file_dir = '../' + output_dir + '/' + dataset_name
    file_path = file_dir + '/'
    # file_path = file_dir + '/' + dataset_name + '_'  # todo: legacy
    if watermark_model_tag != "original":
        file_original_text = open(file_path + 'original.txt', 'r')  # 读原文
        print("open: ", file_path + 'original.txt')

        file_path = file_path + watermark_model_tag
        file_label = open(file_path + '_label.txt', 'r')  # 读标签
        print("open: ", file_path + '_label.txt')
    else:
        file_path = file_dir + 'original'
        file_original_text = None
        file_label = None

    if attack_mode is not None:  # 如果要计算的是被攻击过的文本，则要继续定位
        file_dir = file_dir + '/' + 'attack_' + attack_mode
        file_path = file_dir + '/' + watermark_model_tag + '_' + str(attack_rate)  # str(int(attack_rate * 100))
        # todo: legacy
        # file_path = file_dir + '/' + dataset_name + '_' + watermark_model_tag + '_' + str(attack_rate) + attack_mode

    file_watermarked = open(file_path + '.txt', 'r')
    print("open: ", file_path + '.txt')
    # 记录分数
    file_text_quality = open(file_path + '_' + text_quality_tag + '_' + sim_model_name + '.txt', 'w')  # todo
    print("create: ", file_path + '_' + text_quality_tag + '_' + sim_model_name + '.txt')

    if text_quality_tag == 'meteor':
        _calculate_meteor(file_original_text, file_watermarked, file_label, file_text_quality, attack_mode)
    elif text_quality_tag == 'similarity':

        # model_root = source_root + "roberta-large-mnli"
        # evaluation_tokenizer = RobertaTokenizer.from_pretrained(model_root)
        # evaluation_model = RobertaForSequenceClassification.from_pretrained(model_root).to(device)

        # if attack_mode:
        # _calculate_similarity_for_corrupted(file_original_text, file_watermarked, file_text_quality)
        # _calculate_similarity_for_corrupted_v2(file_original_text, file_watermarked, file_text_quality)
        # else:
        # _calculate_similarity(file_original_text, file_watermarked, file_label, file_text_quality)

        # model_root = source_root + "facebook/bart-large-cnn"
        # bart_score_checkpoint = "../checkpoints/bart_score.pth"
        # bart_scorer = BARTScorer(device='cuda:0', checkpoint=model_root)
        # bart_scorer.load(path=bart_score_checkpoint)
        # _calculate_similarity_by_bart_score(bart_scorer, file_original_text, file_watermarked, file_text_quality)

        # model_root = source_root + "FacebookAI/roberta-large"
        # bert_scorer = BERTScorer(model_root, 17, lang='en')
        # _calculate_similarity_by_bert_score(bert_scorer, file_original_text, file_watermarked, file_text_quality)
        # _calculate_similarity_by_bert_score_v2(bert_scorer, file_original_text, file_watermarked, file_text_quality)

        model_root = source_root + "sentence-transformers/" + sim_model_name
        st_scorer = SentenceTransformer(model_root)
        _calculate_similarity_by_sentran(st_scorer, file_original_text, file_watermarked, file_text_quality)

    elif text_quality_tag == 'ppl':
        model_root = source_root + "openai-community/gpt2-medium"
        evaluation_tokenizer = GPT2Tokenizer.from_pretrained(model_root)
        evaluation_model = GPT2LMHeadModel.from_pretrained(model_root).to(device)

        calculate_ppl(evaluation_tokenizer, evaluation_model, file_watermarked, file_text_quality)

    file_watermarked.close()
    file_text_quality.close()
    if watermark_model_tag != "original":
        file_original_text.close()
        file_label.close()

    print('-' * 32)
    return True


def calculate_text_quality_score_average(
        dataset_name: str, watermark_model_tag: str, text_quality_tag: str,
        attack_mode=None, attack_rate=None, output_dir="ciwater_output"
) -> float:
    file_dir = '../' + output_dir + '/' + dataset_name
    file_path = file_dir + '/' + watermark_model_tag  # todo: output
    # file_path = file_dir + '/' + dataset_name + '_' + watermark_model_tag  # todo: output_legacy

    file_label = open(file_path + '_label.txt', 'r')

    if attack_mode is not None:
        file_dir = file_dir + '/attack_' + attack_mode
        # todo: output
        file_path = file_dir + '/' + watermark_model_tag + '_' + str(attack_rate)  # str(int(attack_rate * 100))
        # todo: output_legacy
        # file_path = file_dir + '/' + dataset_name + '_' + watermark_model_tag + '_' + str(attack_rate)  #str(int(attack_rate * 100))

    # todo: output_legacy::text_quality_tag[0:3]
    # todo: bart
    file_sim = open(file_path + '_' + text_quality_tag + '_all-MiniLM-L6-v2.txt', 'r')

    sim_score = file_sim.readline()
    line_label = file_label.readline()
    temp_value = 0.0
    count = 0
    while line_label:
        if int(line_label.strip('\n')) == 1:
            temp_value = temp_value + float(sim_score.strip('\n'))
            count = count + 1
        sim_score = file_sim.readline()
        line_label = file_label.readline()
    value = temp_value / count
    return value


def draw_text_quality_score_same_source(
        dataset_name: str, watermark_model_tag_list: list[str], text_quality_tag='similarity',  # todo: list
        attack_mode=None, attack_rate=0.1, output_dir="ciwater_output", draw=True
):
    plt_title = dataset_name + ' text-quality scores(' + text_quality_tag + ')'
    print(plt_title)
    categories = []
    values = []
    for model_tag in watermark_model_tag_list:
        categories.append(model_tag)
        value = calculate_text_quality_score_average(
            dataset_name, model_tag, text_quality_tag,
            attack_mode=attack_mode, attack_rate=attack_rate, output_dir=output_dir
        )
        values.append(value)
        print(model_tag, value)

    if draw:
        print("x:", categories)
        print("y:", values)
        plt.figure(figsize=(16, 9), dpi=256)
        plt.bar(categories, values)  # 创建柱状图
        for i in range(len(categories)):
            plt.annotate(
                f'{values[i]:.4f}', xy=(categories[i], values[i]),  # {x_list[i]}:
                ha='center', va='bottom', rotation=0
            )
        # 设置图表属性
        plt.title(plt_title)
        plt.ylim([0.50, 1.00])
        plt.xticks(rotation=0)
        plt.xlabel('watermarked method')
        plt.ylabel(text_quality_tag + 'score')
        plt.show()


def draw_text_quality_score_same_source_c(
        dataset_name: str, watermark_model_tag_list: list[str],  # todo: list
        attack_mode: str, attack_rate_list: list[float],
        text_quality_tag: list[str], output_dir="ciwater_output", draw=True
):
    global plt_color_i

    plt_color_i = 0

    plt_title = dataset_name + ' text-quality scores' + ' (' + attack_mode + ' attack)'
    print(plt_title)

    fig = plt.figure(figsize=(16, 9), dpi=128)
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.set_xlabel('Attack Rate')
    ax1.set_xlim([0.00, 1.00])
    ax1.set_xticks(attack_rate_list)
    ax1.spines['left'].set(linewidth=4, linestyle='-')
    ax1.spines['top'].set(linewidth=4, linestyle='-')
    ax1.spines['bottom'].set(linewidth=4, linestyle='-')

    ax1.set_ylabel('F1-Score')
    ax1.set_ylim([0.50, 1.00])

    ax2 = ax1.twinx()
    ax2.set_ylabel('AUC')
    ax2.set_ylim(bottom=1.00, top=0.50)
    ax2.spines['right'].set(linewidth=4, linestyle='-')

    success_dataset = load_dataset(
        "json",
        data_files='../' + output_dir + '/' + dataset_name + '/attack_' + attack_mode + '/' + attack_mode + '_detect_result.jsonl'
    )

    for i in range(len(watermark_model_tag_list)):

        y, y2 = [], []
        for n in range(len(text_quality_tag)):
            t, t2 = [], []
            y.append(t)
            y2.append(t2)

        for n in range(len(text_quality_tag)):
            for j in range(len(attack_rate_list)):
                # value = calculate_text_quality_score_average(
                #     dataset_name, watermark_model_tag_list[i], text_quality_tag[n],
                #     attack_mode=attack_mode, attack_rate=attack_rate_list[j], output_dir=output_dir
                # )

                # 在数据集中筛选出
                f1 = 0.1
                value = 0.11
                for item in success_dataset['train']:
                    if (item['watermark_method'] == watermark_model_tag_list[i]
                            and item['attack_rate'] == attack_rate_list[j]):
                        f1 = item['f1_score']
                        value = item['AUC']
                        break

                # print(
                #     dataset_name, watermark_model_tag_list[i],
                #     attack_rate_list[j], attack_mode, text_quality_tag[n], round(value, 4), 'f1:', f1
                # )

                # plt.annotate(
                #     f'{value:.4f}', xy=(attack_rate_list[j], value),
                #     ha='center', va='bottom', color=plt_color[i]
                # )
                #
                # plt.annotate(
                #     f'{value:.4f}', xy=(attack_rate_list[j], f1),
                #     ha='center', va='bottom', color=plt_color[i]
                # )

                y[n].append(value)
                y2[n].append(f1)

            if watermark_model_tag_list[i].startswith('ciwater'):
                # temp_float = float(watermark_model_tag_list[i][-3:])
                # temp_float = temp_float * 0.1
                # temp_string = 'ours(' + str(temp_float) + ')'
                temp_string = 'Ours'
            else:
                # temp_float = float(watermark_model_tag_list[i][:2])
                temp_string = 'WTGB'

            ax2.plot(
                attack_rate_list, y[n],
                color=plt_color[plt_color_i], lw=4, marker=plt_line_mark[plt_color_i], markersize=SMALL_SIZE,
                label=temp_string + ' AUC', linestyle=plt_line_style[plt_color_i]
            )

            plt_color_i = plt_color_i + 1

            ax1.plot(
                attack_rate_list, y2[n],
                color=plt_color[plt_color_i], lw=4, marker=plt_line_mark[plt_color_i], markersize=SMALL_SIZE,
                label=temp_string + ' F1-Score', linestyle=plt_line_style[plt_color_i]
            )

            plt_color_i = plt_color_i + 1

    ax1.legend(loc="upper right")
    ax2.legend(loc="lower left")
    plt.savefig(plt_title + '.png')
    plt.show()


def draw_text_quality_score_average_c(
        dataset_list: list[str], watermark_model_tag_list: list[str],  # todo: list
        attack_mode: str, attack_rate_list: list[float], text_quality_tag: list[str],
        output_dir="ciwater_output", draw=True
):
    global plt_color_i
    coco1 = []
    for dataset_name in dataset_list:
        plt_title = dataset_name + ' text-quality scores' + ' (' + attack_mode + ' attack)'
        print(plt_title)

        success_dataset = load_dataset(
            "json",
            data_files='../' + output_dir + '/' + dataset_name + '/attack_' + attack_mode + '/' + attack_mode + '_detect_result.jsonl'
        )

        coco2 = []

        for i in range(len(watermark_model_tag_list)):

            y1, y2 = [], []
            for n in range(len(text_quality_tag)):
                t1, t2 = [], []
                y1.append(t1)
                y2.append(t2)

            for n in range(len(text_quality_tag)):
                for j in range(len(attack_rate_list)):
                    # sim_score = calculate_text_quality_score_average(
                    #     dataset_name, watermark_model_tag_list[i], text_quality_tag[n],
                    #     attack_mode=attack_mode, attack_rate=attack_rate_list[j], output_dir=output_dir
                    # )

                    # 在数据集中筛选出
                    f1 = 0.1
                    sim_score = 0.11
                    for item in success_dataset['train']:
                        if (item['watermark_method'] == watermark_model_tag_list[i]
                                and item['attack_rate'] == attack_rate_list[j]):
                            f1 = item['f1_score']
                            sim_score = item['AUC']
                            break

                    # print(
                    #     dataset_name, watermark_model_tag_list[i],
                    #     attack_rate_list[j], attack_mode, text_quality_tag[n], round(sim_score, 4), 'f1', f1
                    # )

                    y1[n].append(sim_score)
                    y2[n].append(f1)

            coco2.append([y1[0], y2[0]])
        coco1.append(coco2)
    print(coco1)

    if draw:
        new_coco1 = []
        for i in range(0, len(watermark_model_tag_list)):
            y1, y2 = [], []
            for j in range(0, len(attack_rate_list)):
                t1 = coco1[0][i][0][j] + coco1[1][i][0][j] + coco1[2][i][0][j] + coco1[3][i][0][j]
                t2 = coco1[0][i][1][j] + coco1[1][i][1][j] + coco1[2][i][1][j] + coco1[3][i][1][j]
                y1.append(t1 / 4)
                y2.append(t2 / 4)
            new_coco1.append([y1, y2])
        print(new_coco1)

        # plt.figure(figsize=(16, 9), dpi=128)
        # plt.xlim([0.00, 0.425])
        # plt.xticks(attack_rate_list)
        # plt.ylim([0.50, 1.00])
        # plt.xlabel('Attack rate')
        # plt.ylabel('Score')

        fig = plt.figure(figsize=(16, 9), dpi=128)
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.set_xlabel('Attack Rate')
        ax1.set_xlim([0.00, 0.425])
        ax1.set_xticks(attack_rate_list)
        ax1.xaxis.set_tick_params(rotation=20)
        ax1.spines['left'].set(linewidth=4, linestyle='-')
        ax1.spines['top'].set(linewidth=4, linestyle='-')
        ax1.spines['bottom'].set(linewidth=4, linestyle='-')
        # ax1.grid(axis='y', linestyle='--')

        ax1.set_ylabel('F1-Score')
        ax1.set_ylim([0.50, 1.00])

        ax2 = ax1.twinx()
        ax2.set_ylabel('AUC')
        ax2.set_ylim(bottom=0.50, top=1.00)
        ax2.spines['right'].set(linewidth=4, linestyle='-')

        ax2.xaxis.set_ticks_position('top')
        ax2.invert_yaxis()

        for i in range(len(watermark_model_tag_list)):
            if watermark_model_tag_list[i].startswith('ciwater'):
                # temp_float = float(watermark_model_tag_list[i][-3:])
                # temp_float = temp_float * 0.1
                # temp_string = 'ours(' + str(temp_float) + ')'
                temp_string = 'Ours'
            else:
                temp_string = 'WTGB'

            ax2.plot(
                attack_rate_list, new_coco1[i][0],
                color=plt_color[plt_color_i], lw=4, markersize=SMALL_SIZE, marker=plt_line_mark[plt_color_i],
                label=temp_string + ' AUC', linestyle=plt_line_style[plt_color_i]  # n
            )
            plt_color_i = plt_color_i + 1

            ax1.plot(
                attack_rate_list, new_coco1[i][1],
                color=plt_color[plt_color_i], lw=4, markersize=SMALL_SIZE, marker=plt_line_mark[plt_color_i],
                label=temp_string + ' F1-Score', linestyle=plt_line_style[plt_color_i]  # n+1
            )
            plt_color_i = plt_color_i + 1

        ax1.legend(loc="upper right")
        ax2.legend(loc="lower left")
        plt.savefig('all datasets ' + attack_mode + ' attack f1 and auc.png')
        plt.show()

    plt_color_i = 0


def calculate_accuracy(predicted_labels, true_labels):
    """
    计算并返回预测准确率。

    参数:
    predicted_labels (list): 预测的标签列表。
    true_labels (list): 真实的标签列表。

    返回:
    accuracy (float): 准确率，范围在0到1之间。
    """
    if len(predicted_labels) != len(true_labels):
        raise ValueError("预测标签和真实标签列表的长度不一致！")

    correct_count = sum(p == t for p, t in zip(predicted_labels, true_labels))
    accuracy = correct_count / len(true_labels)

    return accuracy


def draw_tatakai_same_source_attacked(source, file_tag_list, attack_rate_list, attack_mode):
    plt.figure()  # 创建一个新的图像窗口
    lw = 1  # 设置线宽
    va = ["top", "bottom"]
    for i in range(len(file_tag_list)):
        average_sim_scores, f1_scores, accuracy, average_meteor_scores = [], [], [], []
        for attack_rate in attack_rate_list:  # _nllb
            file_dir = 'attack_' + attack_mode + '/' + source + '_' + file_tag_list[i] + '_' + str(
                attack_rate) + '_'

            # 计算f1分数,和准确率
            roc_dataset = load_dataset(
                "json", data_files=file_dir + 'result.jsonl'
            )
            print("load: ", file_dir + 'result.jsonl')
            temp_y_true, temp_y_pre = [], []
            for item in roc_dataset['train']:
                y_true = item['label'].strip('\n')
                temp_y_true.append(int(y_true))
                y_pre = item['is_wm']
                temp_y_pre.append(int(y_pre))

            f1_scores.append(f1_score(temp_y_true, temp_y_pre))
            accuracy.append(calculate_accuracy(temp_y_pre, temp_y_true))

            # 只读打开被攻击过文件的 similarity score 和 meteor score
            file_sim = open(file_dir + 'sim.txt', 'r')
            print("open: ", file_dir + 'sim.txt')
            file_meteor = open(file_dir + 'meteor.txt', 'r')
            print("open: ", file_dir + 'meteor.txt')

            file_dir = source + '_' + file_tag_list[i] + '_'
            file_label = open(file_dir + 'label.txt', 'r')
            print("open: ", file_dir + 'label.txt')

            sim_score = file_sim.readline()
            meteor_score = file_meteor.readline()
            line_label = file_label.readline()
            temp_sim_score, count, temp_meteor_score = 0, 0, 0  # 累加每一行
            while line_label:
                if int(line_label.strip('\n')) == 1:
                    temp_sim_score = temp_sim_score + float(sim_score)
                    temp_meteor_score = temp_meteor_score + float(meteor_score)
                    count = count + 1
                sim_score = file_sim.readline()
                meteor_score = file_meteor.readline()
                line_label = file_label.readline()

            average_sim_scores.append(temp_sim_score / count)
            average_meteor_scores.append(temp_meteor_score / count)
        # end for attack_rate in attack_rate_list

        # 画线 same tag, all attack rate
        plt.plot(
            attack_rate_list, f1_scores,
            color=plt_color[i], lw=lw, label='f1-score(method=%s)' % file_tag_list[i], linestyle='-'
        )
        plt.plot(
            attack_rate_list, average_sim_scores,
            color=plt_color[i], lw=lw, label='similarity(method=%s)' % file_tag_list[i], linestyle='--'
        )
        plt.plot(
            attack_rate_list, average_meteor_scores,
            color=plt_color[i], lw=lw, label='meteor(method=%s)' % file_tag_list[i], linestyle='-.'
        )
        plt.plot(
            attack_rate_list, accuracy,
            color=plt_color[i], lw=lw, label='accuracy(method=%s)' % file_tag_list[i], linestyle=':'
        )

        for j in range(len(attack_rate_list)):
            plt.text(
                attack_rate_list[j], f1_scores[j],
                f'{round(f1_scores[j], 2)}', ha="center", va=va[i], color=plt_color[i]
            )
            plt.text(
                attack_rate_list[j], average_sim_scores[j],
                f'{round(average_sim_scores[j], 2)}', ha="center", va=va[i], color=plt_color[i]
            )
            plt.text(
                attack_rate_list[j], average_meteor_scores[j],
                f'{round(average_meteor_scores[j], 2)}', ha="center", va=va[i], color=plt_color[i]
            )
            plt.text(
                attack_rate_list[j], accuracy[j],
                f'{round(accuracy[j], 2)}', ha="center", va=va[i], color=plt_color[i]
            )

    plt.xlim([0.00, 0.90])  # word_level = 0.45
    plt.ylim([0.40, 1.00])
    plt.xlabel("Attack Rate")
    plt.ylabel("score")
    plt.title(source + " (" + attack_mode + ")")
    # 在图表右下角添加图例
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    source_root = "/home/haojifei/develop_tools/transformers/models/"

    source_list = ["medicine", "wiki_csai", "reddit_eli5", "open_qa"]
    # "758083", "808083", "858083" "ciwater5020", "ciwater5015", "ciwater5010"
    file_tag_list = ["808083", "ciwater5010"]

    # for source in source_list:
    #     for file_tag in file_tag_list:
    #         # step1: 计算相似度
    #         calculate_text_quality_score(
    #             source, file_tag, text_quality_tag='similarity', sim_model_name='all-MiniLM-L6-v2'
    #         )
    #         # step2: 计算meteor分数
    #         calculate_text_quality_score(source, file_tag, text_quality_tag='meteor')

    # # step3: 画图
    # for source in source_list:
    #     draw_text_quality_score_same_source(source, file_tag_list, text_quality_tag='similarity', draw=False)
    #     draw_text_quality_score_same_source(source, file_tag_list, text_quality_tag='meteor')

    # step4: 计算ppl分数
    # file_tag_list = ['original', '808083', 'ciwater5010']
    # # model_root = source_root + "openai-community/gpt2-medium"
    # # evaluation_tokenizer = GPT2Tokenizer.from_pretrained(model_root)
    # # evaluation_model = GPT2LMHeadModel.from_pretrained(model_root).to(device)
    # output_root = '../ciwater_output'
    # for source in source_list:
    #     cal_file = []
    #     for file_tag in file_tag_list:
    #         temp_path = output_root + '/' + source + '/' + file_tag
    #         # file_input = open(temp_path + '.txt', 'r')
    #         # file_output = open(temp_path + '_ppl_gpt2-medium.txt', 'w')
    #         # calculate_ppl(evaluation_tokenizer, evaluation_model, file_input, file_output)  # 计算
    #         file_output = open(temp_path + '_ppl_gpt2-medium.txt', 'r')
    #         cal_file.append(file_output)
    #     cal_tag = ['original', 'WTGB', 'Ours']
    #     draw_ppl(cal_file, cal_tag, source)

    # step5: 计算ppl
    # file_tag_list = file_tag_list + ["original"]
    # for source in source_list:
    #     for file_tag in file_tag_list:
    #         calculate_text_quality_score(source, file_tag, text_quality_tag='ppl')

    # step6: 计算被 word-level 攻击过句子的相似度和meteor分数
    # attack_list = ["delete", "substitute"]
    # rate_list = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # , 0.3, 0.5
    # for source in source_list:
    #     for method_tag in file_tag_list:
    #         for rate in rate_list:
    #             for attack in attack_list:
    #                 calculate_text_quality_score(
    #                     source, method_tag, text_quality_tag='similarity', sim_model_name='all-MiniLM-L6-v2',
    #                     attack_mode=attack, attack_rate=rate
    #                 )
    #                 calculate_text_quality_score(
    #                     source, tag, text_quality_tag='meteor',
    #                     attack_mode=attack, attack_rate=rate
    #                 )

    # step7: 画被 word-level 攻击过句子的相似度和meteor分数(屎山：f1)
    text_quality = ['similarity']  # todo: abandon 'meteor'
    # for source in source_list:
    #     for attack in attack_list:
    #         draw_text_quality_score_same_source_c(source, file_tag_list, attack, rate_list, text_quality)

    # step7.1: 画在四个数据集上平均的句子的相似度和f1分数
    # for attack in attack_list:
    #     draw_text_quality_score_average_c(source_list, file_tag_list, attack, rate_list, text_quality, draw=True)

    # step8: 计算被 sentence-level 攻击过句子的相似度和meteor分数(legacy)
    source_list = ["medicine"]
    file_tag_list = ["808083", "ciwater"]
    attack_list = ["translation"]  # todo: "polish", "translation"
    rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for source in source_list:
    #     for method_tag in file_tag_list:
    #         for rate in rate_list:
    #             for attack in attack_list:
    #                 calculate_text_quality_score(
    #                     source, method_tag, text_quality_tag='similarity',
    #                     attack_mode=attack, attack_rate=rate, output_dir="ciwater_output_legacy"
    #                 )

    for source in source_list:
        for attack in attack_list:
            draw_text_quality_score_same_source_c(source, file_tag_list, attack, rate_list, text_quality,
                                                  output_dir="ciwater_output_legacy")

    # draw_text_quality_score_average_c(source_list, file_tag_list, 'polish', rate_list, text_quality,
    #                                   output_dir="ciwater_output_legacy")
    # for source in source_list:
    #     for attack in attack_list:
    #         draw_tatakai_same_source_attacked(source, tag_list, rate_list, attack)
