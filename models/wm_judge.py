from abc import ABC, abstractmethod
import os
import torch

from sentence_transformers import SentenceTransformer


class WMJudge(ABC):
    def __init__(
            self, dataset_name: str, watermark_model_name: str,
            output_dir='./eval_output',
            eval_model_name=None, eval_model_root=None,
            text_quality_tag=None,
            original_set_path=None,
            test_set_path=None, test_set_label_path=None, output_text_quality_set_path=None,
            attack_type=None, attack_rate=0.1
    ):
        self.watermark_model_type = ['black-box', 'white-box']
        self.watermark_type = ['zero-bit', 'multi-bit']

        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

        self._dataset_name = dataset_name
        self._working_dir = self._output_dir + '/' + self._dataset_name
        os.makedirs(self._working_dir, exist_ok=True)

        if original_set_path is None:
            self._original_set_path = self._working_dir + '/original.txt'  # 原文
        else:
            self._original_set_path = original_set_path

        self._watermark_model_name = watermark_model_name
        self._working_dir = self._working_dir + '/' + self._watermark_model_name
        os.makedirs(self._working_dir, exist_ok=True)

        self._working_path = self._working_dir + '/'

        if text_quality_tag is None:
            self._text_quality_tag = 'sim'
        else:
            self._text_quality_tag = text_quality_tag  # todo:check

        if test_set_path is None:
            self._test_set_path = self._working_path + 'mix.txt'  # 混合
        else:
            self._test_set_path = test_set_path

        if test_set_label_path is None:
            self._test_set_label_path = self._working_path + 'label.txt'  # 标签
        else:
            self._test_set_label_path = test_set_label_path

        if output_text_quality_set_path is None:
            self._output_text_quality_set_path = self._working_path + self._text_quality_tag + '.txt'  # 结果输出
        else:
            self._output_text_quality_set_path = output_text_quality_set_path

        self.attack_type = attack_type
        self.attack_rate = attack_rate
        if self.attack_type is not None:
            os.makedirs(self._working_dir + '/' + self.attack_type, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._eval_model_name = eval_model_name
        self._eval_model_root = eval_model_root
        self._eval_model = None
        if self._eval_model_root is not None:
            self.update_eval_model(self._eval_model_name, self._eval_model_root)

    def update_coco(self, dataset_name: str, watermark_model_name: str,):
        pass

    def _update_original_set_path(self):
        self._original_set_path = self._working_dir + '/original.txt'

    def _update_working_path(self):
        self._working_path = self._working_dir + '/' + self._watermark_model_name

    def update_watermark_model_name(self, watermark_model_name: str):
        self._watermark_model_name = watermark_model_name
        self._update_working_path()

    def _update_working_dir(self):
        self._working_dir = self._output_dir + '/' + self._dataset_name
        os.makedirs(self._working_dir, exist_ok=True)
        self._update_working_path()

    def update_dataset(self, dataset_name: str):
        if dataset_name != self._dataset_name:
            self._dataset_name = dataset_name
            self._update_working_dir()

    def update_attack_type(self, attack_type):
        if attack_type != self.attack_type:
            self.attack_type = attack_type
            os.makedirs(self._working_dir + '/' + self.attack_type, exist_ok=True)

    @abstractmethod
    def update_eval_model(self, eval_model_name, eval_model_root):
        pass

    @abstractmethod
    def calculate_similarity(self):
        pass

    def calculate_text_quality_score_average(self, use_label=True) -> float:
        self._update_working_dir(self._dataset_name)

        file_label = open(self._test_set_label_path, 'r')
        self.update_attack_type(self.attack_type)
        file_scores = open(self._output_text_quality_set_path, 'r')

        line_score = file_scores.readline()
        line_label = file_label.readline()
        sum_scores = 0.0
        count = 0
        while line_label:
            if int(line_label.strip('\n')) == 1:
                sum_scores = sum_scores + float(line_score.strip('\n'))
                count = count + 1
            line_score = file_scores.readline()
            line_label = file_label.readline()
        value = sum_scores / count
        print('all:', count, 'average:', value)
        return value


class CustomerJudge(WMJudge):
    def __init__(
            self
    ):
        super().__init__()

    def update_eval_model(self, eval_model_name, eval_model_root):
        self._eval_model_name = eval_model_name
        self._eval_model_root = eval_model_root
        self._eval_model = SentenceTransformer(self._eval_model_root)

    def calculate_similarity(self):
        original_set = open(self._original_set_path, 'r')
        original = original_set.readlines()
        mix_set = open(self._test_set_path, 'r')
        mix = mix_set.readlines()

        embeddings_ori = self._eval_model.encode(original)
        # print(embeddings_ori.shape)

        embeddings_mix = self._eval_model.encode(mix)
        # print(embeddings_wmd.shape)

        sim_scores = self._eval_model.similarity(embeddings_ori, embeddings_mix)
        # print(sim_scores)

        output_text_quality_set = open(self._output_text_quality_set_path, 'w')
        s = 0
        for i in range(len(sim_scores)):
            # print(sim_scores[i][i])
            s += sim_scores[i][i]
            output_text_quality_set.write(str(float(sim_scores[i][i])) + '\n')

        print(s / len(sim_scores))


if __name__ == '__main__':
    print([2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199])
