from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import random
from scipy.stats import norm
from tqdm import tqdm


class WatermarkModel(ABC):
    def __init__(
            self, language: str, watermark_message_type: str,
            use_z_test: bool, z_test_alpha: float
    ):
        self.language = language
        self.watermark_message_type = watermark_message_type
        if watermark_message_type == 'zero-bit' or watermark_message_type == '0-bit':
            pass
        elif watermark_message_type == 'multi-bit':
            pass
        else:
            print("else?")
            pass

        self.use_z_test = use_z_test
        if use_z_test:
            self.z_test_alpha = z_test_alpha
            self.z_alpha = norm.ppf(1 - z_test_alpha, loc=0, scale=1)
            self.p = 0.5

    @abstractmethod
    def watermark_text_generator(self, text: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def watermark_text_detector(self, text: str) -> Dict[str, Any]:
        pass

    def watermark_text_generator_batch(
            self, texts: list[str], positive_proportion: float = 1.0, random_seed: int = 2049
    ) -> list[Dict[str, Any]]:
        random.seed(random_seed)

        texts_len = len(texts)

        if positive_proportion >= 1.0:
            positive_num = texts_len
        elif positive_proportion <= 0.0:
            positive_num = 0
        else:
            positive_num = int(positive_proportion * texts_len)
        mask = [1] * positive_num + [0] * (texts_len - positive_num)
        random.shuffle(mask)
        print("Watermark Mask:", mask)

        results = []
        print(f"Generating watermarked texts... (batch size: {texts_len})")
        bar = tqdm(total=texts_len)
        for i in range(texts_len):
            add_info = {
                "label": mask[i],
                "text": texts[i]
            }
            if mask[i] == 1:
                r = self.watermark_text_generator(texts[i])
                # 弹出 k = "watermark_text" 的 v, 更新至 k = "text" 的 v
                add_info["text"] = r.pop("watermarked_text")
                add_info.update(r)  # 追加剩余信息
            results.append(add_info)
            bar.update(1)
        return results

    def watermark_text_detector_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        results = []
        print(f"Detecting texts... (batch size: {len(texts)})")
        bar = tqdm(total=len(texts))
        for text in texts:
            r = self.watermark_text_detector(text)
            results.append(r)
            bar.update(1)
        return results

    def watermark_text_generator_file(
            self, input_file: str, output_file: str,
            positive_proportion: float = 1.0, random_seed: int = 2049
    ) -> int:

        input_file_ext = input_file.split('.')[-1]

        if input_file_ext == 'jsonl':
            # todo: load jsonl
            lines = []
        else:
            file_input = open(input_file, 'r', encoding='utf-8')
            lines = file_input.readlines()
            file_input.close()

        results = self.watermark_text_generator_batch(lines, positive_proportion, random_seed)

        output_file_ext = output_file.split('.')[-1]
        if output_file_ext != 'jsonl':
            output_file = output_file + '.jsonl'

        file_output = open(output_file, 'w', encoding='utf-8')
        for r in results:
            file_output.write(json.dumps(r) + '\n')
        file_output.close()

        return len(results)

    def watermark_text_detector_file(self, input_file: str, output_file: str):
        """
        Args:
            input_file: 水印文本
                if input_file == 'XXX.jsonl':
                    then 被探测的文本的在jsonl中的 'k' 应当为 'text'
                else:
                    应当保证文件的每一行都是一条文本
            output_file: 水印探测信息输出文件
        Returns:
            写入 output_file 的行数
        """
        input_file_ext = input_file.split('.')[-1]

        file_input = open(input_file, 'r', encoding='utf-8')
        if input_file_ext == 'jsonl':
            lines: list[str] = []
            j_lines = file_input.readlines()
            for j_line in j_lines:
                j = json.loads(j_line)
                lines.append(j['text'])
        else:
            lines: list[str] = file_input.readlines()
            file_input.close()

        results = self.watermark_text_detector_batch(lines)

        output_file_ext = output_file.split('.')[-1]
        if output_file_ext != 'jsonl':
            output_file = output_file + '.jsonl'

        file_output = open(output_file, 'w', encoding='utf-8')
        for r in results:
            file_output.write(json.dumps(r) + '\n')
        file_output.close()

        return len(results)


class WatermarkModelForExistingText(WatermarkModel, ABC):
    def __init__(
            self, language, watermark_message_type: str,
            use_z_test: bool = True, z_test_alpha: float = 0.05
    ):
        super().__init__(language, watermark_message_type, use_z_test, z_test_alpha)
        # todo: more


class WatermarkModelForLLMs(WatermarkModel, ABC):
    def __init__(
            self, language, watermark_message_type: str = '0-bit',
            use_z_test: bool = True, z_test_alpha: float = 0.05
    ):
        super().__init__(language, watermark_message_type, use_z_test, z_test_alpha)
        # todo: more
