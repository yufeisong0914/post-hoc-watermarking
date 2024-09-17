"""
To avoid inserting arbitrary words that make little sense,
we remove all function words, proper nouns, and infrequent rare words.
This refined set forms our final vocabulary, V.

Specifically, we restrict V to only include lowercase nouns[小写名词], verbs[动词], adjectives[形容词],
and adverbs[副词] that occur at least 1,000 times in the WikiText-103 training split.
This results in a final vocabulary of 3,266 words.
"""
from datasets import load_dataset
from collections import Counter
import spacy
from tqdm import tqdm
from openai import OpenAI
import re


def get_embedding(client: OpenAI, text: str) -> list[float]:
    # client = OpenAI(
    #     api_key='sk-d9GXiBuj1oYhyrqF4963E03d62Da4cF3A646E648E509A700',
    #     base_url='https://chatapi.onechats.top/v1/'
    # )
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding


def stats_words(
        dataset_name: str, dataset_root: str, dataset_split: str,
        output_file_path: str, output_file_path_all: str = None,
        word_threshold: int = 1000
):
    """
    统计数据集中单词的频次
    Args:
        dataset_name: 数据集名称
        dataset_root: 数据集路径
        dataset_split: <train, validation, test>
        output_file_path: 统计结果输出路径（经过 word_threshold 过滤后）
        output_file_path_all: 统计结果输出路径（原始统计结果）
        word_threshold: 单词最少出现次数（小于该次数的单词将被过滤）
    Returns: None
    """
    nlp = spacy.load('en_core_web_sm')

    all_tag_list, all_tag_list_num = [], []

    pos_white_list = ['NOUN', 'VERB', 'ADJ', 'ADV']
    pos_white_list_num = [92, 100, 84, 86]

    pos_black_list = ['ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'PROPN', 'SCONJ', 'SYM', 'X',
                      'SPACE', 'PUNCT']
    pos_black_list_num = [103, 96, 87, 90, 97, 89, 95, 85, 93, 94, 98, 101, 99, 91]

    dataset = load_dataset(dataset_root + "/" + dataset_name, split=dataset_split)
    print(dataset)

    jump_line_number = 0
    bar = tqdm(total=len(dataset))
    word_counts = Counter()
    # tag_counts = Counter()
    for instance in dataset:
        text = instance['text']
        if text == '' or text == '\n' or '=' in text:
            jump_line_number += 1
            continue
        else:
            doc = nlp(text)
            words = []
            for token in doc:
                # if token.pos_ in pos_white_list:
                #     if token.pos not in pos_white_list_num:
                #         pos_white_list_num.append(token.pos)
                #     words.append(token.text.lower())
                #
                # if token.pos_ in pos_black_list:
                #     if token.pos not in pos_black_list_num:
                #         pos_black_list_num.append(token.pos)
                #
                # if token.pos_ not in pos_black_list and token.pos_ not in pos_white_list:
                #     print('omit: ', token.text, token.pos_, token.pos)
                #
                # if token.tag_ not in all_tag_list:
                #     all_tag_list.append(token.tag_)
                #     all_tag_list_num.append(token.tag)
                if token.pos in pos_white_list_num:
                    # 过滤单个字符单词，如'a'
                    if len(token.text) > 2:
                        if token.pos == 92:  # 如果是名词
                            if token.text.isupper():  # 过滤大写字母开头名词
                                continue
                            else:
                                words.append(token.text)
                        else:
                            words.append(token.text)
            word_counts.update(words)
        bar.update(1)

    print('skipping lines: ', jump_line_number)
    # print('all white pos: ', pos_white_list_num, ', all black pos: ', pos_black_list_num)
    # print('all tags: ', all_tag_list, all_tag_list_num)

    # 指定文件名
    file_all = open(output_file_path_all, 'w')
    file_filter = open(output_file_path, 'w')
    for word, count in word_counts.items():
        file_all.write(word + ':' + str(count) + '\n')
        if count > word_threshold:
            file_filter.write(word + ':' + str(count) + '\n')
    file_all.close()
    file_filter.close()


def cal_words_embedding(input_file_path: str, output_file_path: str):
    """
    获得单词的词嵌入
    Args:
        input_file_path: 输入 <单词>:<出现次数>
        output_file_path: 输出 <单词>:<词嵌入>
    Returns: None
    """
    file_input = open(input_file_path, 'r')
    file_output = open(output_file_path, 'w')

    client = OpenAI(
        api_key='sk-d9GXiBuj1oYhyrqF4963E03d62Da4cF3A646E648E509A700',
        base_url='https://chatapi.onechats.top/v1/'
    )

    lines = file_input.readlines()
    bar = tqdm(total=len(lines))
    for line in lines:
        # line = line.strip('\n')
        temp_word_count = line.split(':')
        # print(temp_word_count)
        embedding = get_embedding(client, temp_word_count[0])
        file_output.write(temp_word_count[0] + ':' + str(embedding) + '\n')
        bar.update(1)
    file_input.close()
    file_output.close()


def extract_floats_from_string(s):
    # 使用正则表达式匹配所有的浮点数
    float_pattern = r"[-+]?\d*\.\d+"
    floats = re.findall(float_pattern, s)

    # 将提取到的字符串转换成浮点数
    float_list = [float(num) for num in floats]

    return float_list


def seesee_words_embedding(file_path: str):
    file = open(file_path, 'r')
    lines = file.readlines()
    for line in lines:
        embeddings = extract_floats_from_string(line)
        print(len(embeddings), embeddings)
        # line = file.readline()
    file.close()


def seesee_eql():
    s1 = ['one p one = one', ' = p two two', 'p three three']
    for i in s1:
        if '=' in i:
            print(i)


if __name__ == '__main__':
    # step 1: 统计出现次数>1000的单词，得到secret table
    stats_words(
        "wikitext-103-raw-v1",
        "/home/haojifei/develop_tools/transformers/datasets/Salesforce/wikitext",
        "train",
        './wikitext-103-raw-v1_train_1000_v2.txt',
        './wikitext-103-raw-v1_train_all_v2.txt',
        word_threshold=1000
    )
    # step 2: 获得每个单词的嵌入向量
    # cal_words_embedding('./wikitext-103-raw-v1_train_1000.txt', './wikitext-103-raw-v1_train_1000_embed.txt')
    # step 3:
    # seesee_words_embedding('./wikitext-103-raw-v1_train_1000_embed.txt')
