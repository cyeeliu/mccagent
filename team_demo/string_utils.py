"""字符串工具库 - 初始版本（朴素实现）。

功能：
  1. 统计单词数量
  2. 反转字符串（按单词）
  3. 检查回文
  4. 去除重复字符
"""


def count_words(text):
    """统计字符串中的单词数量 - 朴素方法：按空格分割后计数。"""
    if text is None or len(text) == 0:
        return 0
    words = text.split(" ")
    # 过滤掉空字符串
    result = []
    for w in words:
        if len(w) > 0:
            result.append(w)
    return len(result)


def reverse_words(text):
    """按单词反转字符串 - 朴素方法：分割、反转、拼接。"""
    if text is None or len(text) == 0:
        return ""
    words = text.split(" ")
    # 过滤空字符串
    filtered = []
    for w in words:
        if len(w) > 0:
            filtered.append(w)
    # 反转列表
    reversed_list = []
    for i in range(len(filtered) - 1, -1, -1):
        reversed_list.append(filtered[i])
    # 拼接
    result = ""
    for i in range(len(reversed_list)):
        if i > 0:
            result = result + " "
        result = result + reversed_list[i]
    return result


def is_palindrome(text):
    """检查是否为回文 - 朴素方法：逐字符比较。"""
    if text is None:
        return False
    # 转小写并移除空格
    cleaned = ""
    for ch in text:
        if ch != " ":
            cleaned = cleaned + ch.lower()
    if len(cleaned) == 0:
        return False
    # 逐字符比较
    n = len(cleaned)
    for i in range(n // 2):
        if cleaned[i] != cleaned[n - 1 - i]:
            return False
    return True


def remove_duplicate_chars(text):
    """去除重复字符，保留首次出现 - 朴素方法：逐字符检查。"""
    if text is None or len(text) == 0:
        return ""
    result = ""
    for ch in text:
        found = False
        for c in result:
            if c == ch:
                found = True
                break
        if not found:
            result = result + ch
    return result
