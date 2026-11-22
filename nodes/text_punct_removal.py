# -*- coding: utf-8 -*-
# Created by My: 按标点移除前后内容节点
# 说明：
# - 输入一段文本、一个标点（支持正则）和索引，移除该标点前或后的内容（包含标点本身）
# - index >= 0：删除「开头 → 第 index 个命中标点（含标点）」
# - index < 0：删除「倒数第 |index| 个命中标点（含标点）→ 结尾」
# - 关闭 enable 时源输入与移除结果都等于源输入；输入为 None 时原样透传不报错

import re


class TextPunctRemoveMs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "要处理的文本，可由上游连接传入；为 None 时原样透传。"}),
                "punctuation": ("STRING", {"default": ",", "tooltip": "标点或分隔符；use_regex 开启时按正则解析，例如 , 。 [,，] \\s+；留空则原样输出。"}),
                "index": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1, "tooltip": "正数=左起第 N 个命中标点（0 为第一个），删除它及之前的内容；负数=右起第 N 个（-1 为最后一个），删除它及之后的内容；越界或无匹配时原样输出。"}),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "关闭后不解析标点，源输入与移除结果都等于源输入。"}),
                "use_regex": ("BOOLEAN", {"default": True, "tooltip": "开启时标点按正则解析；若标点本身是 . ( ) [ ] * + ? | 等符号，建议关闭以按字面量匹配。"}),
                "strip_result": ("BOOLEAN", {"default": True, "tooltip": "开启时仅对移除后的句子清理首尾空白，源输入始终原样输出。"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("source_text", "result_text")
    FUNCTION = "remove"
    CATEGORY = "A_my_nodes/text"
    DESCRIPTION = "按指定标点（支持正则）和索引，移除标点前或后的内容（含标点本身），同时输出源输入。"

    def remove(self, text, punctuation, index=0, enable=True, use_regex=True, strip_result=True):
        # 输入为 None 时原样透传，避免中断工作流
        if text is None:
            return (None, None)

        # 非字符串类型不做处理，直接透传
        if not isinstance(text, str):
            return (text, text)

        # 关闭开关或文本为空时，两路输出都等于源输入
        if not enable or text == "":
            return (text, text)

        # 标点为空时无法定位，原样输出
        if punctuation is None or punctuation == "":
            return (text, text)

        pattern = punctuation if use_regex else re.escape(punctuation)
        try:
            matches = list(re.finditer(pattern, text))
        except re.error as e:
            print(f"⚠️ [TextPunctRemoveMs] 标点正则无效，已原样输出：{e}")
            return (text, text)

        # 没有任何命中，原样输出
        if not matches:
            return (text, text)

        if index >= 0:
            # 正索引：从左数第 index 个命中，删除它及之前的内容
            if index >= len(matches):
                print(f"⚠️ [TextPunctRemoveMs] index={index} 超出命中数量 {len(matches)}，已原样输出。")
                return (text, text)
            match = matches[index]
            result = text[match.end():]
        else:
            # 负索引：从右数第 |index| 个命中，删除它及之后的内容
            position = len(matches) + index
            if position < 0:
                print(f"⚠️ [TextPunctRemoveMs] index={index} 超出命中数量 {len(matches)}，已原样输出。")
                return (text, text)
            match = matches[position]
            result = text[:match.start()]

        if strip_result:
            result = result.strip()

        return (text, result)
