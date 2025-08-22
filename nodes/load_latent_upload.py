# -*- coding: utf-8 -*-
# Created time : 2024/12/19
# Author : Assistant
# File   : load_latent_upload.py
# Description : LoadLatent节点的上传版本，支持文件上传和拖拽功能

import os
import hashlib
import safetensors.torch
import folder_paths


class LoadLatentUpload:
    """支持上传功能的LoadLatent节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                if f.endswith(".latent"):
                    files.append(f)
        
        return {
            "required": {
                "latent": (sorted(files),),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "loaders"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    DESCRIPTION = "加载latent文件，支持文件上传和拖拽功能"

    def load(self, latent):
        """加载latent文件"""
        latent_path = folder_paths.get_annotated_filepath(latent)
        latent_data = safetensors.torch.load_file(latent_path, device="cpu")
        
        # 处理版本兼容性
        multiplier = 1.0
        if "latent_format_version_0" not in latent_data:
            multiplier = 1.0 / 0.18215
            
        samples = {"samples": latent_data["latent_tensor"].float() * multiplier}
        return (samples,)

    @classmethod
    def IS_CHANGED(s, latent):
        """检查文件是否发生变化"""
        latent_path = folder_paths.get_annotated_filepath(latent)
        m = hashlib.sha256()
        with open(latent_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        """验证输入参数"""
        if not folder_paths.exists_annotated_filepath(latent):
            return "Invalid latent file: {}".format(latent)
        return True


# 节点映射
NODE_CLASS_MAPPINGS = {
    "LoadLatentUpload": LoadLatentUpload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLatentUpload": "Load Latent (Upload) 加载Latent文件(支持上传) by My",
}