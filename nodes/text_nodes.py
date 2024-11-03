# -*- coding: utf-8 -*-
# Created time : 2024/10/05 22:05 
# Auther : ygh
# File   : text_nodes.py
# Description :
import time
import os
import re


# import torch
# import cv2
# import numpy as np
# import os


def str_to_list(s):
    lst = eval(s)
    return lst


class CoordinateTessPosNeg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "coordinates_negative": ("STRING", {"forceInput": True}),
            }
        }

    CATEGORY = "My_node/text"
    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "example_func"

    def example_func(self, coordinates_positive, coordinates_negative):
        start_time = time.time()
        coordinates_positive = str_to_list(coordinates_positive)
        coordinates_negative = str_to_list(coordinates_negative)
        len_pos = len(coordinates_positive)
        len_neg = len(coordinates_negative)
        if len_pos < len_neg:
            coordinates_negative = coordinates_negative[:len_pos]
        if len_pos == 0:
            coordinates_positive = None
        else:
            coordinates_positive = str(coordinates_positive)
        if len_neg == 0:
            coordinates_negative = None
        else:
            coordinates_negative = str(coordinates_negative)

        print(f"sam2pos_neg cost time: {time.time() - start_time} s")
        return (coordinates_positive, coordinates_negative)

class FilterClothingWords:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"forceInput": True}),
                "additional_keywords": ("STRING", {"forceInput": False}),
            },
            "optional": {
                "add_to_keywords_file": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "My_node/text"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_string",)
    FUNCTION = "filter_clothing_words"

    def __init__(self):
        self.clothing_keywords = self.load_clothing_keywords()

    def load_clothing_keywords(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        keywords_file = os.path.join(current_dir, 'clothing_keywords.txt')
        with open(keywords_file, 'r') as f:
            return set(word.strip().lower() for word in f.read().split(','))

    def filter_clothing_words(self, input_string, additional_keywords=None, add_to_keywords_file=False):
        start_time = time.time()
        
        # 使用正则表达式分割输入字符串，支持逗号和句号
        words = re.split(r'[,.]', input_string)
        
        # 处理附加关键词
        if additional_keywords:
            additional_keywords_set = set(word.strip().lower() for word in additional_keywords.split(','))
            self.clothing_keywords.update(additional_keywords_set)  # 合并到现有关键词集合
            
            # 如果选择添加到文件且附加关键词不为空，则写入文件
            if add_to_keywords_file:
                self.write_keywords_to_file(additional_keywords_set)

        # 过滤掉包含服饰关键词的单词或短语
        filtered_words = [word.strip() for word in words if not any(keyword in word.lower() for keyword in self.clothing_keywords)]
        
        # 将过滤后的单词或短语重新组合成字符串
        result = ', '.join(filtered_words)
        
        print(f"filter_clothing_words cost time: {time.time() - start_time} s")
        return {"ui": {"text": result}, "result": (result,)}
    
    def write_keywords_to_file(self, keywords):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        keywords_file = os.path.join(current_dir, 'clothing_keywords.txt')
        with open(keywords_file, 'a') as f:  # 以追加模式打开文件
            for keyword in keywords:
                f.write(f"{keyword}, ")
