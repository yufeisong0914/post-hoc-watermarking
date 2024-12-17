import re
import spacy
from nltk.tokenize import sent_tokenize


def preprocess_string(input_string):
    input_string = input_string.replace("\ n", " ")
    input_string = input_string.replace("\\n", " ")
    input_string = input_string.replace("\n", " ").replace("\t", " ").replace("\r", " ")

    # 把多个连续的空格替换为一个空格
    input_string = re.sub('( +)', ' ', input_string)

    # I 'm -> I'm
    input_string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", input_string)

    input_string = input_string.replace("s \'", "s\'")  # text = re.sub("s '", "s'", text)
    # print(text)

    # 666 . 666 -> 666.666 或 123 , 345 , 678 -> 123,456,789
    # input_string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", input_string)

    # U . S . -> U.S.
    input_string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", input_string)

    # 去掉标点符号左边的空格
    input_string = re.sub("( )([,\.!?:;)])", r"\2", input_string)

    # 去掉左括号(右边的空格: '( ' -> '('
    input_string = re.sub("([(])( )", r"\1", input_string)

    # reduce both space
    input_string = re.sub("( )(')( )(\S+)( )(')", r"\2\4\6", input_string)
    input_string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", input_string)
    input_string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", input_string)
    input_string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", input_string)

    input_string = input_string.replace("\\n", " ")
    input_string = input_string.replace(" ' ", "'")

    return input_string.strip()


def preprocess_string2(input_string):
    input_string = input_string.replace("\",", ",\"")
    input_string = input_string.replace("\".", ".\"")
    return input_string


def preprocess_string3(input_string):
    input_string = re.sub("(\d+)([,.])( )(\d+)", r"\1\2\4", input_string)
    return input_string


def preprocess_string4(input_string):
    input_string = re.sub("( )(\d+)([.])( )(\w)", r"\2\3\5", input_string)
    return input_string


def remove_adjacent_commas(input_string):
    # 使用正则表达式替换连续的数字逗号
    output_string = re.sub(r'(\d),(\d)', r'\1\2', input_string)
    return output_string


if __name__ == "__main__":

    root = '/home/haojifei/develop_things/nlp_projects/parals-watermarking/ciwater_output/'
    source_list = ["reddit_eli5"]  # todo:"reddit_eli5", "open_qa"
    file_tag_list = ["858083", "808083", "758083", "ciwater5010", "ciwater5015", "ciwater5020"]
    for source in source_list:
        file_dir = root + source + '/'
        for file_tag in file_tag_list:
            file = open(file_dir + file_tag + "_2.txt", "r")
            file2 = open(file_dir + file_tag + "_3.txt", "w")
            line = file.readline()
            while line:
                new_line = preprocess_string4(line)
                file2.write(new_line)
                line = file.readline()
