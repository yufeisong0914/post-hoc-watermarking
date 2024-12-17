import re

def rm_blank_lines(text: str) -> str:
    out_put = text.replace("\n", " ")
    out_put = re.sub('( +)', ' ', out_put)
    return out_put