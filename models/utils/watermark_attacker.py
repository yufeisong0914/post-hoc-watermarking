from abc import ABC, abstractmethod

import openai


class WatermarkAttacker(ABC):
    def __init__(self, language, model_name):
        self.language = language
        self.model_name = model_name
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def attack_token(self, text: str) -> str:
        pass

    @abstractmethod
    def attack_doc(self, text: str) -> str:
        pass


class OpenAIAttacker(WatermarkAttacker):
    def __init__(
            self, language: str,
            model_name: str, model_api_key: str, model_api_base_url: str, prompt_template: str
    ):
        self.model_api_key = model_api_key
        self.model_api_base_url = model_api_base_url
        self.prompt_template = prompt_template
        super().__init__(language, model_name)

    def _load_model(self):
        if self.model_api_base_url:
            client = openai.OpenAI(
                api_key=self.model_api_key,
                base_url=self.model_api_base_url,
            )
        else:
            client = openai.OpenAI(api_key=self.model_api_key)
        return client

    def attack_token(self, text: str) -> str:
        return text

    def attack_doc(self, text: str) -> str:
        message = self.prompt_template.format(text)
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": message
            }]
        )
        corrupted_text = response.choices[0].message.content
        return corrupted_text
