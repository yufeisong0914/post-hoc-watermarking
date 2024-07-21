from datasets import load_dataset
from datasets import load_from_disk

datasets_root = "/home/haojifei/develop_tools/transformers/datasets"

"""
**Datasets.**【数据集】为了评估我们的方法，我们主要利用\[10]中的人类 ChatGPT 比较语料库 (HC3)。 
HC3 数据集为检查人类书写和 ChatGPT 生成的中文和英文文本的语言和文体特征提供了重要资源。
我们选择 ChatGPT 答案进行评估。具体来说，我们从以下每个英语子类别中收集了 200 个样本：
wiki_csai、open_qa、medicine 和 reddit_eli5。每个英语样本的长度为200±5个单词。我们总共获得了 800 个样本作为我们的英语数据集。

同样，我们从相应的中文子类别中选择了 800 个样本，包括 baike、open_qa、medicine 和 nlpcc_dbqa。
每个中文样本的长度为200±5个字符。需要注意的是，200个汉字和200个英文单词的信息承载能力是不同的，中文和英文语言模型之间存在性能差异。
因此，这两种语言的结果将表现出细微的差异。
"""

def filter_chatgpt_answers(item):
    for i in range(len(item["chatgpt_answers"])):
        if len(item["chatgpt_answers"][i].split()) >= 200:
            return True
    return False


def filter_chatgpt_answers_length(item):
    return len(item["unwatermarked"]) > 1  # chatgpt_answers


def filter_unwatermarked(item):
    if item["unwatermarked"]:
        return True
    else:
        return False

def update_unwatermarked(item):
    unwatermarked = "ciwater"
    length = 0
    for i in range(len(item["unwatermarked"])):  # chatgpt_answers
        if len(item["unwatermarked"][i]) > length:
            length = len(item["unwatermarked"][i])
            unwatermarked = item["unwatermarked"][i]

    if unwatermarked == "ciwater":
        print("finding a void line")
    return {"unwatermarked": unwatermarked}


def update_column_unwatermarked(item):
    return {"unwatermarked": " ".join(item['unwatermarked'].split()[:205])}


def process_dataset_v0():
    hc3 = load_dataset(datasets_root + "/Hello-SimpleAI/HC3", name="all")
    # print(hc3)
    '''
    DatasetDict({
        train: Dataset({
            features: ['id', 'question', 'human_answers', 'chatgpt_answers', 'source'],
            num_rows: 24322
        })
    })
    '''
    hc3 = hc3.remove_columns('question')
    hc3 = hc3.remove_columns('human_answers')
    # 过滤掉['chatgpt_answers']为空的
    hc3_sample = hc3.filter(filter_unwatermarked)
    hc3_sample = hc3_sample.rename_column(original_column_name='chatgpt_answers', new_column_name='unwatermarked')
    # print(hc3_sample)
    # 在一条样本中,['unwatermarked']这一列可能有多条，选择最长的一条
    hc3_sample = hc3_sample.map(update_unwatermarked)
    # print(hc3_sample)
    # print(hc3_sample['train'][:5])
    # 取205个token
    hc3_sample = hc3_sample.map(update_column_unwatermarked)
    hc3_sample.save_to_disk(datasets_root + "/Hello-SimpleAI/processed/hc3_sample")


def process_dataset_v1(source_name_list):
    for source_name in source_name_list:
        raw_hc3_each = load_dataset(datasets_root + "/Hello-SimpleAI/HC3", name=source_name)
        # print(hc3)
        """
        "medicine"
        DatasetDict({
            train: Dataset({
                features: ['id', 'question', 'human_answers', 'chatgpt_answers'],
                num_rows: 1248
            })
        })
        "open_qa"
        DatasetDict({
            train: Dataset({
                features: ['id', 'question', 'human_answers', 'chatgpt_answers'],
                num_rows: 1187
            })
        })
        "reddit_eli5"
        DatasetDict({
            train: Dataset({
                features: ['id', 'question', 'human_answers', 'chatgpt_answers'],
                num_rows: 17112
            })
        })
        "wiki_csai"
        DatasetDict({
            train: Dataset({
                features: ['id', 'question', 'human_answers', 'chatgpt_answers'],
                num_rows: 842
            })
        })
        """
        raw_hc3_each = raw_hc3_each.remove_columns('question')
        raw_hc3_each = raw_hc3_each.remove_columns('human_answers')
        raw_hc3_each = raw_hc3_each.rename_column(
            original_column_name='chatgpt_answers', new_column_name='unwatermarked'
        )
        raw_hc3_each.save_to_disk(datasets_root + "/Hello-SimpleAI/processed/hc3_" + source_name + "_v1")


def process_dataset_v2(source_name_list):
    for source_name in source_name_list:
        hc3_each = load_from_disk(datasets_root + "/Hello-SimpleAI/processed/hc3_" + source_name + "_v1")
        print(hc3_each)
        hc3_each = hc3_each.filter(filter_unwatermarked)
        hc3_each_v2 = hc3_each.map(update_unwatermarked)
        print(hc3_each_v2)
        flag = hc3_each_v2.save_to_disk(datasets_root + "/Hello-SimpleAI/processed/hc3_" + source_name + "_v2")
        print(flag)


if __name__ == "__main__":
    # process_dataset_v0()
    # process_dataset_v0_step2()

    process_dataset_v1(["medicine", "open_qa", "reddit_eli5", "wiki_csai"])
    process_dataset_v2(["medicine", "open_qa", "reddit_eli5", "wiki_csai"])
