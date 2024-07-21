from datasets import load_from_disk
from models.watermark_faster import watermark_model
import json
from tqdm import tqdm
import os
import ciwater_util_string


def generate_original_text(source, file_tag, output_dir="ciwater_output"):
    my_dataset = load_from_disk(datasets_root + "/Hello-SimpleAI/processed/hc3_" + source + "_v2")
    os.makedirs(output_dir, exist_ok=True)  # 设置exist_ok=True，使得即使目标文件夹已经存在，程序也不会抛出FileExistsError异常
    os.makedirs(output_dir + "/" + source, exist_ok=True)
    output_file_dir = output_dir + '/' + source + '/' + file_tag + '.txt'
    file = open(output_file_dir, 'w')
    print("creat: ", output_file_dir)
    bar = tqdm(total=200)
    count = 0
    for text in my_dataset['train'].shuffle(seed=2049).select(range(220)):
        if 100 <= len(text['unwatermarked']) <= 2000:
            # temp = text['unwatermarked']
            # print(temp)
            # temp = ciwater_util_string.preprocess_string(temp)
            # print(temp)
            file.write(ciwater_util_string.preprocess_string(text['unwatermarked']) + "\n")
            count = count + 1
            bar.update(1)
            if count == 200:
                break
    file.close()


def watermark_extract(raw, alpha, detect_mode):
    """
    Args:
        raw: 文本[string]
        alpha: 上α分位点[0.00]
        detect_mode: ['precise','fast']
    Returns: 置信度
    """
    is_watermark, p_value, n, ones, z_value = model.watermark_detector(raw, alpha, detect_mode)
    confidence = (1 - p_value) * 100
    return f"{confidence:.2f}%"


def generate_mixed(watermark_model_class, dataset_name: str, watermark_model_tag: str, output_dir="ciwater_output"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + "/" + dataset_name, exist_ok=True)

    file_dir = output_dir + '/' + dataset_name

    file_path = file_dir + '/'
    file = open(file_path + 'original.txt', 'r')
    print("open: ", file_path + 'original.txt')
    line = file.readline()

    file_path = file_path + watermark_model_tag
    file_mix = open(file_path + '.txt', 'w')
    print("creat: ", file_path + '.txt')
    file_label = open(file_path + '_label.txt', 'w')
    print("creat: ", file_path + '_label.txt')
    file_i = open(file_path + '_watermark_index.txt', 'w')
    print("creat: ", file_path + '_watermark_index.txt')

    use_our_method = False
    if watermark_model_tag[:7] == "ciwater":
        use_our_method = True

    i, wm_c, unwm_c = 0, 0, 0
    bar = tqdm(total=200)
    coco = 1
    while line:
        if i % 2 == 0:  # 加水印
            if use_our_method:
                watermarked_text, watermark_token_index = watermark_model_class.embed_v1(line)
                file_i.write(str(watermark_token_index) + "\n")
            else:
                watermarked_text = watermark_model_class.embed(line)
            file_mix.write(ciwater_util_string.preprocess_string(watermarked_text) + "\n")
            file_label.write(str(1) + "\n")
            wm_c += 1
        else:  # 不加水印
            file_mix.write(line)
            if use_our_method:
                file_i.write("[]\n")
            file_label.write(str(0) + "\n")
            unwm_c += 1
        i = i + 1
        line = file.readline()
        bar.update(1)
        coco = coco + 1

    file.close()
    file_mix.close()
    file_label.close()
    file_i.close()

    file_mix_count = open(file_path + '_count.json', 'w')
    file_mix_count.write("{\"wm\":" + str(wm_c) + ",\"unwm\":" + str(unwm_c) + "}\n")
    file_mix_count.close()


def generate_json_v2(label, original_is_watermark, original_p_value, original_n, original_ones, original_z_value):
    # 构建结果字典
    result = {
        'label': label,
        'is_wm': '1' if original_is_watermark else '0',
        'p_score': original_p_value,
        'ones/n': str(original_ones) + '/' + str(original_n),
        'z_value': original_z_value
    }
    # 将结果转换为 JSON 格式
    return json.dumps(result)


def detect_mixed(
        my_model, dataset_name: str, watermark_model_tag: str,
        output_dir="ciwater_output", attack_mode=None, attack_rate=None
):
    os.makedirs(output_dir, exist_ok=True)
    file_dir = output_dir + '/' + dataset_name
    os.makedirs(file_dir, exist_ok=True)

    file_path = file_dir + '/' + watermark_model_tag

    file_label = open(file_path + '_label.txt', 'r')
    print("open: ", file_path + '_label.txt')

    if attack_mode:
        file_dir = file_dir + '/attack_' + attack_mode
        os.makedirs(attack_mode, exist_ok=True)
        file_path = file_dir + '/' + watermark_model_tag + '_' + str(attack_rate)

    file = open(file_path + '.txt', 'r')
    print("open: ", file_path + '.txt')

    file_result = open(file_path + '_result.jsonl', 'w')
    print("create: ", file_path + '_result.jsonl')

    file_result_index = open(file_path + '_result_index.txt', 'w')
    print("create: ", file_path + '_result_index.txt')

    file_result_words = open(file_path + '_result_words.txt', 'w')
    print("create: ", file_path + '_result_words.txt')

    line = file.readline()  # 读取第一行
    line_label = file_label.readline()  # 读取第一行
    bar = tqdm(total=200)
    while line:
        is_watermark, p_value, n, ones, z_value, mask_words_index, mask_words = my_model.watermark_detector(line)
        file_result.write(str(generate_json_v2(line_label, is_watermark, p_value, n, ones, z_value)) + '\n')
        file_result_index.write(str(mask_words_index) + '\n')
        file_result_words.write(str(mask_words) + '\n')

        line = file.readline()
        line_label = file_label.readline()
        # print(is_watermark, p_value, n, ones, z_value)
        bar.update(1)

    file.close()
    file_label.close()
    file_result.close()


if __name__ == "__main__":
    datasets_root = "/home/haojifei/develop_tools/transformers/datasets"
    # tau_word_list = [0.75, 0.8, 0.85]
    tau_sent = 0.8
    lamda = 0.83
    model = watermark_model(
        models_dir="/home/haojifei/develop_tools/transformers/models",
        w2v_dir="/home/haojifei/develop_tools/w2v_models",
        language="English", detect_mode="precise", alpha=0.05,
        tau_word=0.8, tau_sent=tau_sent, lamda=lamda
    )

    # todo: "medicine", "wiki_csai", "reddit_eli5", "open_qa"
    source_list = ["medicine", "wiki_csai", "reddit_eli5", "open_qa"]

    # step1: 从4个数据集分别抽取200个没加水印的原文
    # for source in source_list:
    #     generate_original_text(source, "original")

    # step2: 使用BBTW方法，对其中的一半句子加水印，生成混合数据集并探测
    # for tau_word in tau_word_list:
    #     model.tau_word = tau_word
    #     file_tag = str(int(tau_word * 100)) + str(int(tau_sent * 100)) + str(int(lamda * 100))
    #     for source_name in source_list:
    #         generate_mixed(model, source_name, file_tag)  # 生成
    #         detect_mixed(model, source_name, file_tag)  # 探测

    # step3: 使用我们的方法，对其中的一半句子加水印，生成混合数据集并探测
    # replacement_rate_list = [9.9]  # 1.0, 1.5, 2.0,
    # model.detect_mode = "ciwater"
    # for replacement_rate in replacement_rate_list:
    #     model.replacement_rate = replacement_rate
    #     print("set replacement_rate: " + str(replacement_rate))
    #     file_tag = "ciwater50" + str(int(replacement_rate * 10))
    #     for source in source_list:
    #         generate_mixed(model, source, file_tag, output_dir="ciwater_output")  # 生成
    #         detect_mixed(model, source, file_tag)  # 探测

    # step4: 测一测被攻击过的文本
    model.detect_mode = "precise"
    tau_word_list = [0.8]  # 0.75, 0.8, 0.85
    # tau_word_list = [0.8]
    rate_list = [0.3, 0.5]  # 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5
    attack_method_list = ["delete", "substitute"]
    for source in source_list:
        for tau_word in tau_word_list:
            model.tau_word = tau_word
            file_tag = str(int(tau_word * 100)) + str(int(tau_sent * 100)) + str(int(lamda * 100))
            for rate in rate_list:
                for attack_method in attack_method_list:
                    detect_mixed(model, source, file_tag, attack_mode=attack_method, attack_rate=rate)

    # step4: 测一测被攻击过的文本
    # model.detect_mode = "ciwater"
    # replacement_rate_list = [1.0]  # 1.0, 1.5, 2.0
    # rate_list = [0.3, 0.5]  # 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5
    # attack_method_list = ["delete", "substitute"]  # "delete", "substitute"
    # for source in source_list:
    #     for replacement_rate in replacement_rate_list:
    #         model.replacement_rate = replacement_rate
    #         file_tag = "ciwater50" + str(int(replacement_rate * 10))
    #         for rate in rate_list:
    #             for attack_method in attack_method_list:
    #                 detect_mixed(model, source, file_tag, attack_mode=attack_method, attack_rate=rate)

    # todo: re-translation
    # rate_list = [0.5, 0.6, 0.7, 0.8, 0.9] # 0.1, 0.2, 0.3, 0.4,
    # subset_list = ["translation"]
    # for source in source_list:
    #     for rate in rate_list:
    #         for subset in subset_list:
    #             detect_mixed(model, source, 'ciwater', "ours", rate, subset)
    # for source in source_list:
    #     for rate in rate_list:
    #         for subset in subset_list:
    #             detect_mixed(model, source, '808083', "precise", rate, subset)

    # text = "There can be many causes of nervousness and lack of confidence, and it's important to note that these feelings are common and normal to experience at times. However, if they are persistent and interfere with your daily life, it may be helpful to speak with a mental health professional. They can help you identify any underlying issues and provide you with strategies to manage your symptoms. It's possible that you may be experiencing anxiety or social anxiety, which can cause nervousness and lack of confidence in social situations. Other potential causes of these symptoms could include past trauma or adverse experiences, low self-esteem, or certain life stressors. It's also worth noting that crying in response to emotions is a normal and healthy way to express and process feelings. However, if you feel that your crying is excessive or disrupting your daily life, it may be helpful to speak with a mental health professional to explore possible causes and develop coping strategies. It's important to remember that it's okay to seek help if you are struggling with your emotions or if they are impacting your daily life. A mental health professional can provide you with support and guidance to help you manage your symptoms and improve your overall well-being."
    # 没加水印 测一测
    # print("-" * 64)
    # is_watermark, p_value, n, ones, z_value = model.watermark_detector(text, 0.05, 'ciwater')
    # confidence = (1 - p_value) * 100
    # print(str(generate_json_v2(0, is_watermark, p_value, n, ones, z_value)), confidence)
    # 加上水印
    # print("-" * 64)
    # watermarked_text = model.embed(text)
    # print(watermarked_text)
    # # 测一测
    # print("-" * 64)
    # is_watermark, p_value, n, ones, z_value = model.watermark_detector(watermarked_text, 0.05, 'ciwater')
    # confidence = (1 - p_value) * 100
    # print(str(generate_json_v2(1, is_watermark, p_value, n, ones, z_value)), confidence)
