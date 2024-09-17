from abc import ABC, abstractmethod
from typing import Dict, Any
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

    def watermark_text_generator_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        watermarked = []
        bar = tqdm(total=len(texts))
        for text in texts:
            watermarked_text = self.watermark_text_generator(text)
            watermarked.append(watermarked_text)
            bar.update(1)
        return watermarked

    def watermark_text_generator_file(
            self, input_file: str, input_file_ext: str, output_file: str,
            positive_proportion: float = 1.0, random_seed: int = 2049
    ) -> int:
        # todo: load different types of files
        if input_file_ext in ['json', 'jsonl']:
            # todo
            return 0
        else:
            file_input = open(input_file, 'r', encoding='utf-8')
            file_output = open(output_file, 'w', encoding='utf-8')
            lines = file_input.readlines()
            generator_results = self.watermark_text_generator_batch(lines)
            for result in generator_results:
                file_output.write(str(result) + '\n')
            file_output.close()
            file_input.close()
            return len(generator_results)

    def watermark_text_detector_dataset(self, texts: list[str]) -> list[str]:
        pass


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
