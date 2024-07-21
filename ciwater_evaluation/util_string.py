import re


def preprocess_string(input_string):
    input_string = input_string.replace("\ n", " ")
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

    return input_string.strip()


def remove_adjacent_commas(input_string):
    # 使用正则表达式替换连续的数字逗号
    output_string = re.sub(r'(\d),(\d)', r'\1\2', input_string)
    return output_string


if __name__ == "__main__":
    # text = "yes    i wanna fuck : you have --- pussy ( like oranges ) in size 22 . 22, it 's yours ' hhh, ' NO '!."
    # print(text)
    # print(preprocess_string(text))
    a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    print(len(a))
    flo = 3.44
    print(round(flo))
    flo = 3.54
    print(round(flo))
