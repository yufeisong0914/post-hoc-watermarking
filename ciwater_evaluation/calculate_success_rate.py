from datasets import load_dataset
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import json

datasets_root = "/home/haojifei/develop_tools/transformers/datasets"
# plt_color = ['red', 'green', 'blue', 'gold', 'violet', 'grey']

plt_color = ['blue', 'red', 'gold', 'blue', 'green', 'red']

# 设置全局字体大小
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

# 设置各个部分的字体大小
rcParams['font.size'] = BIGGER_SIZE  # 默认字体大小
rcParams['axes.titlesize'] = BIGGER_SIZE  # 标题字体大小
rcParams['axes.labelsize'] = MEDIUM_SIZE  # 坐标轴标签字体大小
rcParams['xtick.labelsize'] = MEDIUM_SIZE  # x轴刻度标签字体大小
rcParams['ytick.labelsize'] = MEDIUM_SIZE  # y轴刻度标签字体大小
rcParams['legend.fontsize'] = BIGGER_SIZE  # 图例字体大小
rcParams['figure.titlesize'] = BIGGER_SIZE  # 图形标题字体大小


def draw_roc_same_tag(source_list, file_tag):
    all_label, all_z_value = [], []
    for source in source_list:
        file_name = source + "_" + file_tag + "_result.jsonl"
        roc_dataset = load_dataset("json", data_files=file_name)
        print(roc_dataset)

        temp_label, temp_z_value = [], []
        for item in roc_dataset['train']:
            l = item['label'].strip('\n')
            temp_label.append(int(l))  # 提取标签
            z_value = round(item['z_value'], 2)  # 保存两位小数
            temp_z_value.append(z_value)  # 提取z值

        all_label.append(temp_label)
        all_z_value.append(temp_z_value)

    plt.figure()  # 创建一个新的图像窗口
    line_width = 1  # 设置线宽
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR(False Positive Rate)")
    plt.ylabel("TPR(True Positive Rate)")
    plt.title(file_tag + ' ROC')

    # 每个数据集
    for i in range(len(source_list)):
        y = np.array(all_label[i])
        scores = np.array(all_z_value[i])
        fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
        area_under_the_curve = auc(fpr, tpr)
        line_label = source_list[i] + ' ROC curve (area = ' + str(round(area_under_the_curve, 2)) + ')'
        plt.plot(fpr, tpr, color=plt_color[i], lw=line_width, label=line_label)

    # 汇总
    all_label_temp, all_z_value_temp = [], []
    for i in range(len(source_list)):
        all_label_temp = all_label_temp + all_label[i]
        all_z_value_temp = all_z_value_temp + all_z_value[i]

    y = np.array(all_label_temp)
    scores = np.array(all_z_value_temp)
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
    area_under_the_curve = auc(fpr, tpr)
    line_label = 'all ROC curve (area = ' + str(round(area_under_the_curve, 2)) + ')'
    plt.plot(fpr, tpr, color='black', lw=line_width + 1, label=line_label)

    # 对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
    plt.legend(loc="lower right")
    plt.show()


def calculate_success_rate(file_path: str, success_rate_tag='accuracy-score'):
    roc_dataset = load_dataset("json", data_files=file_path)
    print('load:', file_path)

    if success_rate_tag == 'roc':
        all_true_label, all_z = [], []
        for item in roc_dataset['train']:
            x = item['label'].strip('\n')
            all_true_label.append(int(x))
            y = round(item['z_value'], 6)
            all_z.append(y)  # 提取z值
        fpr, tpr, thresholds = roc_curve(all_true_label, all_z, pos_label=1)  # 计算
        area_under_the_curve = auc(fpr, tpr)  # 计算
        return area_under_the_curve, fpr, tpr, thresholds

    else:  # success_rate_tag == 'accuracy' or success_rate_tag == 'f1-score'
        all_true_label, all_pred_label = [], []
        for item in roc_dataset['train']:
            x = item['label'].strip('\n')
            all_true_label.append(int(x))  # true label
            y = item['is_wm']
            all_pred_label.append(int(y))  # pred label

        # 屎山
        cm = confusion_matrix(all_true_label, all_pred_label)
        TN, FP, FN, TP = cm.ravel()
        FPR = FP / (FP + TN)  # 假正率
        TPR = TP / (TP + FN)  # 真正率
        TNR = TN / (TN + FP)  # 真负率，也叫特异度
        FNR = FN / (FN + TP)  # 假阴率
        print("False Positive Rate (FPR):", FPR)
        print("True Positive Rate (TPR):", TPR)
        print("True Negative Rate (TNR):", TNR)
        print("False Negative Rate (FNR):", FNR)

        if success_rate_tag == 'f1-score':
            temp_f1_score = f1_score(all_true_label, all_pred_label)  # 计算
            return temp_f1_score, 0.0, 0.0, 0.0
        else:
            temp_accuracy_score = accuracy_score(all_true_label, all_pred_label)  # 计算
            return temp_accuracy_score, 0.0, 0.0, 0.0


def calculate_success_rate_corrupt(
        file_path: str, dataset_name: str, watermark_model_tag: str, attack_mode=None, attack_rate=None
):
    roc_dataset = load_dataset("json", data_files=file_path)
    print('load:', file_path)

    all_true_label, all_pred_label, all_z = [], [], []
    for item in roc_dataset['train']:
        true_label = item['label'].strip('\n')  # true label
        all_true_label.append(int(true_label))

        pred_label = item['is_wm']
        all_pred_label.append(int(pred_label))  # pred label

        z = round(item['z_value'], 6)  # 提取z值
        all_z.append(z)

    # 屎山
    cm = confusion_matrix(all_true_label, all_pred_label)
    TN, FP, FN, TP = cm.ravel()
    FPR = FP / (FP + TN)  # 假正率
    TPR = TP / (TP + FN)  # 真正率
    TNR = TN / (TN + FP)  # 真负率，也叫特异度
    FNR = FN / (FN + TP)  # 假阴率

    temp_accuracy_score = accuracy_score(all_true_label, all_pred_label)  # 计算
    temp_f1_score = f1_score(all_true_label, all_pred_label)  # 计算

    fpr, tpr, thresholds = roc_curve(all_true_label, all_z, pos_label=1)
    area_under_the_curve = auc(fpr, tpr)

    # 创建一个字典
    data = {
        "dataset_name": dataset_name,
        "watermark_method": watermark_model_tag,
        "attack_method": attack_mode,
        "attack_rate": attack_rate,
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "TP": int(TP),
        "FPR": float(FPR),
        "TNR": float(TNR),
        "TPR": float(TPR),
        "FNR": float(FNR),
        "accuracy": float(temp_accuracy_score),
        "f1_score": round(float(temp_f1_score), 4),
        "AUC": float(area_under_the_curve),
        # "ROC": {
        #     "AUC": float(area_under_the_curve),
        #     "fpr": tuple(fpr.tolist()),
        #     "tpr": tuple(tpr.tolist()),
        #     "thresholds": tuple(thresholds.tolist()),
        # },
    }

    # 将字典转换为JSON格式的字符串
    json_data = json.dumps(data, ensure_ascii=False)

    # 打印JSON字符串
    # print(json_data)

    return json_data


def coco(
        dataset_name: str, watermark_model_tags: list[str], success_rate_tag='accuracy-score',
        attack_mode=None, attack_rate_list=None, output_dir='ciwater_output', draw=True
) -> None:
    """
    Args:
        dataset_name:
        watermark_model_tags:
        success_rate_tag:
        attack_mode:
        attack_rate_list: ['roc', 'f1-score', 'accuracy-score']
        output_dir:
        draw:
    Returns:
        void
    """

    success_rate_tag = success_rate_tag.lower()
    # 图片标题
    plt_title = dataset_name + ' ' + success_rate_tag
    # 工作目录
    file_dir = '../' + output_dir + '/' + dataset_name
    t1, t2, t3, t4 = [], [], [], []
    if attack_mode is None:
        print('=' * 32, '\n', plt_title, '\n', '=' * 32)

        # file_all_result = open(file_dir + '/all_success_rate_results.jsonl', 'w')

        for tag in watermark_model_tags:
            file_path = file_dir + '/' + tag + '_result.jsonl'
            # json_data = calculate_success_rate2(file_path)
            # print(tag, ':', json_data, '\n', '-' * 32)
            z1, z2, z3, z4 = calculate_success_rate(file_path, success_rate_tag)
            print(tag, success_rate_tag + ':', z1)
            t1.append(z1)
            t2.append(z2)
            t3.append(z3)
            t4.append(z4)
    else:
        if attack_rate_list is None:
            exit(-1)
        file_dir = file_dir + '/attack_' + attack_mode  # 切换工作目录

        file_detect_result = open(file_dir + '/' + attack_mode + '_detect_result.jsonl', 'w')

        for attack_rate in attack_rate_list:  # 遍历所有的攻击频率
            sub_plt_title = plt_title + ' (' + str(attack_rate) + ' ' + attack_mode + ' attack)'
            print('=' * 32, '\n', sub_plt_title, '\n', '=' * 32)
            for tag in watermark_model_tags:
                # file_path = file_dir + '/' + tag + '_' + str(attack_rate) + "_result.jsonl"
                # todo: legacy, if attack_mode==translation,  + attack_mode
                file_path = file_dir + '/' + dataset_name + '_' + tag + '_' + str(attack_rate) + "_result.jsonl"

                json_data = calculate_success_rate_corrupt(file_path, dataset_name, tag, attack_mode, attack_rate)
                print(json_data, '\n')
                file_detect_result.write(json_data + '\n')
                # z1, z2, z3, z4 = calculate_success_rate(file_path, success_rate_tag)
                # print(tag, str(attack_rate), attack_mode, success_rate_tag, ':', z1)
                # t1.append(z1)
                # t2.append(z2)
                # t3.append(z3)
                # t4.append(z4)
        file_detect_result.close()

    if draw:
        if attack_mode is None:
            if success_rate_tag == 'roc':
                plt.figure(figsize=(8, 5), dpi=256)
                # plt.title(plt_title)
                plt.xlim([0.0, 1.0025])
                plt.ylim([0.8, 1.0025])
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                for i in range(len(watermark_model_tags)):
                    if watermark_model_tags[i].startswith('ciwater'):
                        # temp_float = float(watermark_model_tags[i][-3:])
                        # temp_float = temp_float * 0.1
                        # temp_string = 'ours(' + str(temp_float) + ')'
                        temp_string = 'ours'
                    else:
                        # temp_float = float(watermark_model_tags[i][:2])
                        # temp_float = temp_float * 0.01
                        # temp_string = 'WTGB(' + str(temp_float) + ')'
                        temp_string = 'WTGB'

                    line_label = temp_string + ', AUC=' + str(round(t1[i], 4))
                    plt.plot(t2[i], t3[i], color=plt_color[i], lw=4, label=line_label)
                plt.plot([0, 1], [0, 1], color='navy', lw=4, linestyle='--')  # 对角线
                plt.legend(loc="lower right")
            else:
                plt.figure(figsize=(16, 9), dpi=256)
                plt.title(plt_title)
                plt.ylim([0.50, 1.00])
                plt.xlabel("watermark method")
                plt.ylabel(success_rate_tag)
                plt.bar(watermark_model_tags, t1)  # 柱
                for i in range(len(watermark_model_tags)):
                    plt.annotate(f'{t1[i]:.4f}', xy=(watermark_model_tags[i], t1[i]), ha='center', va='bottom')
            plt.savefig(file_dir + '/' + plt_title + '.png')
            plt.show()
        else:
            if success_rate_tag == 'roc':
                plt.figure(figsize=(8, 8), dpi=256)

                plt.title(plt_title)
                plt.xlim([0.0, 1.05])
                plt.ylim([0.0, 1.05])
                plt.xlabel("FPR(False Positive Rate)")
                plt.ylabel("TPR(True Positive Rate)")
                for i in range(len(watermark_model_tags)):
                    line_label = watermark_model_tags[i] + '(AUC=' + str(round(t1[i], 4)) + ')'
                    plt.plot(t2[i], t3[i], color=plt_color[i], lw=1, label=line_label)
                plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
                plt.legend(loc="lower right")
            else:
                plt.figure(figsize=(16, 9), dpi=256)
                plt.title(plt_title + ' (' + attack_mode + ' attack)')
                plt.xticks(attack_rate_list)
                plt.ylim([0.0, 1.00])
                plt.xlabel("watermark method")
                plt.ylabel(success_rate_tag)
                # for i in range(len(attack_rate_list)):

                plt.bar(watermark_model_tags, t1)
            plt.savefig(file_dir + '/' + plt_title + '.png')
            plt.show()


if __name__ == "__main__":
    # draw_roc_same_tag(["reddit_eli5", "wiki_csai", "medicine", "open_qa"], "858083")
    # draw_roc_same_tag(["reddit_eli5", "wiki_csai", "medicine", "open_qa"], "808083")
    # draw_roc_same_tag(["reddit_eli5", "wiki_csai", "medicine", "open_qa"], "ciwater")

    source_list = ["medicine", "wiki_csai", "reddit_eli5", "open_qa"]
    # "758083", "858083", "808083", "ciwater5020", "ciwater5015", "ciwater5010"
    file_tag_list = ["808083", "ciwater5010"]

    # step1:
    # for source in source_list:
    #     coco(source, file_tag_list, 'roc')
        # coco(source, file_tag_list, 'accuracy-score')
        # coco(source, file_tag_list, 'f1-score')

    # # step2:
    # attack_list = ["delete", "substitute"]  # todo: substitute
    # rate_list = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # for source in source_list:
    #     for attack in attack_list:
    #         coco(source, file_tag_list, 'accuracy-score', attack, rate_list, draw=False)

    source_list = ["medicine"]
    file_tag_list = ["ciwater", "808083"]
    attack_list = ["translation"]  # polish, translation
    rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for source in source_list:
        for attack in attack_list:
            coco(source, file_tag_list, 'accuracy-score', attack, rate_list, output_dir='ciwater_output_legacy', draw=False)
