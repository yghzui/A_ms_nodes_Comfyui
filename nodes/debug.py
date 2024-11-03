import re

# 服饰相关的词语或短语
clothing_terms = [
    r'\b(?:dress|shirt|blouse|pants|trousers|skirt|shorts|jacket|coat|trench coat|blazer|suit|tuxedo|waistcoat|sweater|cardigan|hoodie|turtleneck|jumpsuit|romper|overalls|parka|bomber jacket|windbreaker|raincoat|peacoat|vest|tank top|camisole|crop top|polo shirt|denim jacket|leather jacket|puffer jacket|anorak|kimono|poncho|shawl|cape|tunic|kaftan|nightgown|bathrobe|gown|tracksuit|joggers|leggings|dungarees|bodysuit|leotard|playsuit|coveralls|t-shirt|henley shirt|flannel shirt|dress shirt|peplum top|halter top|tube top|bandeau|slip dress|ball gown|cocktail dress|maxi dress|midi dress|mini dress|wrap dress|sheath dress|shift dress|A-line dress|pencil skirt|pleated skirt|skater skirt|tutu|culottes|chinos|cargo pants|harem pants|bell-bottoms|skinny jeans|bootcut jeans|straight-leg jeans)\b',
    r'\bmortarboard\b',  # 针对例子中提到的特定服饰
    r'\b(?:black|white|red|blue|green|yellow|brown|pink|purple|gray|beige|burgundy)\b'  # 常见颜色
]

# 将服饰词语合并为一个正则表达式模式
clothing_pattern = re.compile('|'.join(clothing_terms), re.IGNORECASE)

def remove_clothing_descriptions(text):
    # 删除与服饰相关的描述
    cleaned_text = re.sub(clothing_pattern, '', text)
    
    # 去除多余的空格
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    
    return cleaned_text

# 示例文本
text = "The image shows a young girl wearing a black graduation gown and a black mortarboard. She is standing in front of a white wall with Chinese characters written on it."

# 调用函数剔除服饰描述
cleaned_text = remove_clothing_descriptions(text)

# 输出结果
print(cleaned_text)
