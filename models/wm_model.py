from abc import ABC, abstractmethod
from typing import Dict, Any


class WatermarkModel(ABC):
    def __init__(self, watermark_message_type: str, use_z_test: bool = True, z_test_alpha: float = 0.05):

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

    @abstractmethod
    def watermark_text_generator(self, text: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def watermark_text_detector(self, text: str) -> Dict[str, Any]:
        pass

    def watermark_text_generator_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        watermarked = []
        for text in texts:
            watermarked_text = self.watermark_text_generator(text)
            watermarked.append(watermarked_text)
        return watermarked

    def watermark_text_generator_file(self, texts: list[str]) -> list[str]:
        pass

    def watermark_text_detector_dataset(self, texts: list[str]) -> list[str]:
        pass


class WatermarkModelForExistingText(WatermarkModel, ABC):
    def __init__(self, watermark_message_type: str, use_z_test: bool = True, z_test_alpha: float = 0.05):
        super().__init__(watermark_message_type, use_z_test, z_test_alpha)
        # todo: more


class WatermarkModelForLLMs(WatermarkModel, ABC):
    def __init__(self, watermark_message_type: str = '0-bit', use_z_test: bool = True, z_test_alpha: float = 0.05):
        super().__init__(watermark_message_type, use_z_test, z_test_alpha)
        # todo: more
