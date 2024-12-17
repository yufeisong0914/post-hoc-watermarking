import openai
import spacy
from nltk.tokenize import sent_tokenize
import util_string
import random
from tqdm import tqdm

openai.api_key = ''
openai.base_url = ''

Instruction = 'Please polish the input text without changing its meaning and structure.'
Input = 'The input text is:'

nlp = spacy.load('en_core_web_sm')


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


def refine_sentence(sentence):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": Input + sentence}
        ],
    )
    # print(response)
    text = response.choices[0].message.content
    return text.strip()


def generate_corrupted(file_path, attack_rate_list):
    # 最大的攻击概率
    attack_rate_max = attack_rate_list[len(attack_rate_list) - 1]

    file = open(file_path, 'r')
    print("open: ", file_path)
    line = file.readline()
    file_name = file_path.strip(".txt")
    # file_label = open(file_name + '_label.txt', 'r')
    # line_label = file_label.readline()

    attacked_files, attacked_files_i, attacked_files_s = [], [], []
    for i in range(len(attack_rate_list)):
        attacked_file_temp = open(file_name + '_' + str(attack_rate_list[i]) + 'polish.txt', 'w')
        print("create: ", file_name + '_' + str(attack_rate_list[i]) + 'polish.txt')
        attacked_files_i_temp = open(file_name + '_' + str(attack_rate_list[i]) + 'polish_i.txt', 'w')
        print("create: ", file_name + '_' + str(attack_rate_list[i]) + 'polish_i.txt')
        attacked_files_s_temp = open(file_name + '_' + str(attack_rate_list[i]) + 'polish_s.txt', 'w')
        print("create: ", file_name + '_' + str(attack_rate_list[i]) + 'polish_s.txt')
        attacked_files.append(attacked_file_temp)
        attacked_files_i.append(attacked_files_i_temp)
        attacked_files_s.append(attacked_files_s_temp)

    sentences = []
    len_sentence = 0
    num_sent_per_line = []
    bar = tqdm(total=200)
    while line:
        # if int(line_label.strip('\n')) == 1:
        doc_paragraph = nlp(util_string.preprocess_string(line))
        current_sentences = [sentence.text for sentence in doc_paragraph.sents]

        sentences = sentences + current_sentences  # 累计句子
        len_sentence = len_sentence + len(current_sentences)  # 累计句子数
        num_sent_per_line.append(len(current_sentences))  # 记录每行文本的句子数

        # 保证被攻击的一段话至少由10个句子组成
        if len_sentence < 10:
            # continue loop
            line = file.readline()
            # line_label = file_label.readline()
            continue
        else:
            attacked_sentence_num = int(len_sentence * attack_rate_max)
            attacked_sentence_index = generate_unique_random_integers(
                attacked_sentence_num, 0, len_sentence, seed=2049
            )
            print("-----", attacked_sentence_index)
            # polishing
            polishing_bar = tqdm(total=attacked_sentence_num)
            attacked_sentences = []
            for index in attacked_sentence_index:
                attacked_sentence = refine_sentence(sentences[index])
                attacked_sentence = util_string.preprocess_string(attacked_sentence)
                attacked_sentences.append(attacked_sentence)
                polishing_bar.update(1)
            # 分别写入
            for i in range(len(attack_rate_list)):
                n = int(attack_rate_list[i] * len_sentence)
                temp_index = attacked_sentence_index[:n]
                bias = 0
                for num_sent in num_sent_per_line:
                    attacked_line = ""
                    i_per_line, s_per_line = [], []
                    for j in range(num_sent):
                        if j + bias in temp_index:
                            # find
                            k = 0
                            for k in range(len(temp_index)):
                                if temp_index[k] == j + bias:
                                    break
                            attacked_sentence = attacked_sentences[k]
                            i_per_line.append(j)
                            s_per_line.append(attacked_sentence)
                        else:
                            attacked_sentence = sentences[j + bias]
                        attacked_line = attacked_line + attacked_sentence + " "
                    # 将处理后的句子写入新的txt文件
                    attacked_files[i].write(attacked_line + '\n')
                    attacked_files_i[i].write(str(i_per_line) + '\n')
                    attacked_files_s[i].write(str(s_per_line) + '\n')
                    # 偏置累加
                    bias = bias + num_sent

            # 重新初始化
            sentences = []
            len_sentence = 0
            num_sent_per_line = []

        # continue loop
        line = file.readline()
        # line_label = file_label.readline()
        bar.update(1)

    file.close()
    # file_label.close()
    for i in range(len(attack_rate_list)):
        attacked_files[i].close()
        attacked_files_i[i].close()
        attacked_files_s[i].close()


def generate_retranslated(source, file_tag, attack_rates):
    file_dir = "attack_translation_nllb/" + source + "_" + file_tag + "_"
    file = open(file_dir + "0.0.txt", 'r')  # 原文
    print("open: ", file_dir + "0.0.txt")
    line = file.readline()
    file_translated = open(file_dir + "g2en.txt", 'r')  # 重译文
    print("open: ", file_dir + "g2en.txt")
    line_translated = file_translated.readline()

    file_translate_attacked_i, file_translate_attacked = [], []
    for attack_rate in attack_rates:
        file_attacked_i = open(file_dir + str(attack_rate) + "_i.txt", 'r')  # 各攻击概率下的攻击位置
        print("open: ", file_dir + str(attack_rate) + "_i.txt")
        file_translate_attacked_i.append(file_attacked_i)
        file_attacked = open(file_dir + str(attack_rate) + ".txt", 'w')  # 生成被攻击过的文本
        print("create: ", file_dir + str(attack_rate) + ".txt")
        file_translate_attacked.append(file_attacked)

    line_i = []
    for i in range(len(attack_rates)):
        temp_line_i = file_translate_attacked_i[i].readline()
        line_i.append(temp_line_i)

    flag = False
    bar = tqdm(total=200)
    coco = 0
    while line:
        coco = coco + 1
        flag = False
        # doc_paragraph = nlp(util_string.preprocess_string(line))
        # current_sentences = [sentence.text for sentence in doc_paragraph.sents]
        current_sentences = sent_tokenize(line)
        csl = len(current_sentences)
        # doc_paragraph_trans = nlp(util_string.preprocess_string(line_translated))
        # current_sentences_trans = [sentence.text for sentence in doc_paragraph_trans.sents]
        current_sentences_trans = sent_tokenize(line_translated)
        cstl = len(current_sentences_trans)

        if csl != cstl:
            doc_paragraph = nlp(util_string.preprocess_string(line))
            current_sentences = [sentence.text for sentence in doc_paragraph.sents]
            csl = len(current_sentences)

            doc_paragraph_trans = nlp(util_string.preprocess_string(line_translated))
            current_sentences_trans = [sentence.text for sentence in doc_paragraph_trans.sents]
            cstl = len(current_sentences_trans)

            if csl != cstl:
                flag = True
                print(coco, ": ", csl, cstl)

        for i in range(len(attack_rates)):

            if flag:
                file_translate_attacked[i].write(" \n")
            else:
                temp_attacked_sentence = ""
                len_line_i = len(line_i[i])
                # print(len_line_i)
                if len_line_i > 3:  # line_i[i] != []\n
                    # 提取将要被修改的句子下标
                    temp_i = line_i[i][1:len_line_i - 2]
                    temp_i_list = temp_i.split(',')
                    # 将字符串转化为整型list
                    temp_i_int_list = []
                    for s in temp_i_list:
                        temp_i_int_list.append(int(s.strip()))
                    # 生成被替换攻击过的句子
                    for j in range(len(current_sentences)):
                        if j in temp_i_int_list:
                            temp_attacked_sentence = temp_attacked_sentence + current_sentences_trans[j] + " "
                        else:
                            temp_attacked_sentence = temp_attacked_sentence + current_sentences[j] + " "
                else:
                    temp_attacked_sentence = line

                file_translate_attacked[i].write(util_string.preprocess_string(temp_attacked_sentence) + "\n")

        line_i = []  # 清空
        for i in range(len(attack_rates)):
            temp_line_i = file_translate_attacked_i[i].readline()
            line_i.append(temp_line_i)
        line = file.readline()
        line_translated = file_translated.readline()
        bar.update(1)

    file.close()
    file_translated.close()
    for i in range(len(attack_rates)):
        file_translate_attacked_i[i].close()
        file_translate_attacked[i].close()


if __name__ == "__main__":
    attack_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # # 调用函数处理txt文件
    # source_list = ["open_qa"]  # , "reddit_eli5", "wiki_csai"
    # method_list = ["ciwater"]  # "808083",
    # for source in source_list:
    #     for method in method_list:
    #         generate_corrupted(source + '_' + method + '.txt', attack_rate_list)

    # generate_retranslated("medicine", "808083", attack_rate_list)
    generate_retranslated("medicine", "ciwater", attack_rate_list)
