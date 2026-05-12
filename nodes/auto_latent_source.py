from nodes import EmptyLatentImage, VAEEncode


class AutoLatentSource:
    """
    根据 mode 选择调用官方 VAEEncode 或 EmptyLatentImage。

    - image: 强制使用图像编码为潜空间
    - empty: 强制生成空潜空间
    - auto: 有 image 时走 image，否则走 empty
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": 16384, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "mode": (["image", "empty", "auto"], {"default": "auto"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "build"
    CATEGORY = "A_my_nodes/latent"
    DESCRIPTION = "Use official VAEEncode or EmptyLatentImage based on the selected mode."

    def build(self, width, height, batch_size, mode, image=None, vae=None):
        resolved_mode = mode
        if mode == "auto":
            resolved_mode = "empty" if image is None else "image"

        if resolved_mode == "image":
            if image is None:
                raise ValueError("mode=image 时必须提供 image。")
            if vae is None:
                raise ValueError("mode=image 时必须提供 vae。")
            encoder = VAEEncode()
            print(f"使用 VAEEncode 编码图像")
            return encoder.encode(vae=vae, pixels=image)

        if resolved_mode == "empty":
            latent_generator = EmptyLatentImage()
            print(f"使用 EmptyLatentImage 生成空潜空间")
            return latent_generator.generate(width=width, height=height, batch_size=batch_size)

        raise ValueError(f"不支持的 mode: {mode}")
