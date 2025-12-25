import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from langchain_core.runnables import Runnable
from pathlib import Path


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_PROCESSOR = None


def _load_model_once():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        print(f"[BLIP] Загрузка модели на {_DEVICE}...")
        # Загружаем модель с логами в консоль
        model_name = open(Path(__file__).parent / "ImageModelfile", encoding='utf-8').readline().strip()
        _PROCESSOR = BlipProcessor.from_pretrained(model_name)
        _MODEL = BlipForConditionalGeneration.from_pretrained(
            model_name,
        ).to(_DEVICE).eval()
        print("[BLIP] Модель загружена.")


class BLIPImageCaptioner(Runnable):
    def __init__(self, max_new_tokens: int = 50):
        self.max_new_tokens = max_new_tokens
        _load_model_once()


    def invoke(self, input, config=None):
        print(input)
        # Поддержка: str (путь), PIL.Image, или dict с 'image'
        if isinstance(input, dict):
            image_input = input.get("image")
            max_tokens = input.get("max_new_tokens", self.max_new_tokens)
        elif isinstance(input, list):
            image_input = input[-1]
            max_tokens = self.max_new_tokens
        else:
            image_input = input
            max_tokens = self.max_new_tokens

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Input must be path (str), PIL.Image, or dict with 'image' key.")

        inputs = _PROCESSOR(image, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            out = _MODEL.generate(**inputs, max_new_tokens=max_tokens)
            caption = _PROCESSOR.decode(out[0], skip_special_tokens=True)
            # print(f"[BLIP] {caption} ({time.time() - start:.2f}s)")
        return caption



_load_model_once()

