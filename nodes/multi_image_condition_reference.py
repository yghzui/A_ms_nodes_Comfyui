import inspect
import logging
import re

from comfy_extras.nodes_edit_model import ReferenceLatent
from nodes import VAEEncode


class MultiImageConditionReference:
    MAX_IMAGES = 12
    IMAGE_INPUT_PATTERN = re.compile(r"^image_(\d+)$")

    @classmethod
    def INPUT_TYPES(cls):
        dynamic_inputs = {
            "image_1": (
                "IMAGE",
                {
                    "optional": True,
                    "tooltip": "第 1 张参考图。连接后前端会继续追加下一个 image 输入接口。",
                },
            ),
        }

        stack = inspect.stack()
        if len(stack) > 2 and stack[2].function == "get_input_info":
            class LimitedImageContainer:
                def __contains__(self, item):
                    match = cls.IMAGE_INPUT_PATTERN.match(str(item))
                    if not match:
                        return False
                    index = int(match.group(1))
                    return 1 <= index <= cls.MAX_IMAGES

                def __getitem__(self, key):
                    return "IMAGE", {
                        "optional": True,
                        "tooltip": f"动态参考图输入 {key}。",
                    }

            dynamic_inputs = LimitedImageContainer()

        return {
            "required": {
                "vae": ("VAE", {"tooltip": "用于将输入图片编码为 latent 的 VAE。"}),
                "positive": ("CONDITIONING", {"tooltip": "输入的正向 conditioning。"}),
                "negative": ("CONDITIONING", {"tooltip": "输入的负向 conditioning。"}),
            },
            "optional": dynamic_inputs,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    OUTPUT_TOOLTIPS = (
        "追加了全部参考图 latent 后的正向 conditioning。",
        "追加了全部参考图 latent 后的负向 conditioning。",
    )
    FUNCTION = "apply_reference_images"
    CATEGORY = "A_my_nodes/conditioning"
    DESCRIPTION = (
        "将多张参考图按顺序通过 VAE 编码为 latent，并依次追加到正向/负向 conditioning 的 "
        "reference_latents 中。未连接图片时直接原样返回 conditioning。"
    )
    _vae_encode = VAEEncode()

    @classmethod
    def _get_sorted_image_keys(cls, kwargs):
        image_keys = []
        for key, value in kwargs.items():
            if value is None:
                continue
            match = cls.IMAGE_INPUT_PATTERN.match(str(key))
            if not match:
                continue
            index = int(match.group(1))
            if 1 <= index <= cls.MAX_IMAGES:
                image_keys.append((index, key))
        image_keys.sort(key=lambda item: item[0])
        return [key for _, key in image_keys]

    def apply_reference_images(self, vae, positive, negative, **kwargs):
        image_keys = self._get_sorted_image_keys(kwargs)
        if not image_keys:
            logging.info("[MultiImageConditionReference] No image input connected, returning original conditioning.")
            return positive, negative

        logging.info(
            "[MultiImageConditionReference] Using %d image inputs: %s",
            len(image_keys),
            ", ".join(image_keys),
        )

        current_positive = positive
        current_negative = negative

        for key in image_keys:
            image = kwargs.get(key)
            if image is None:
                continue

            latent = self._vae_encode.encode(vae, image)[0]
            current_positive = ReferenceLatent.execute(current_positive, latent).result[0]
            current_negative = ReferenceLatent.execute(current_negative, latent).result[0]

        return current_positive, current_negative
