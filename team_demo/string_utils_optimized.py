"""字符串工具库 - 优化版本。

优化要点：
  - 使用 str.split() 自动处理多空格
  - 用切片 [::-1] 替代手动反转循环
  - 用 str.join() 替代循环拼接（避免 O(n²)）
  - 用 set 实现 O(1) 去重查找
  - 添加类型注解和完整文档字符串

功能接口与 string_utils.py 完全一致。
"""

from __future__ import annotations


def count_words(text: str | None) -> int:
    """统计字符串中的单词数量。

    优化：使用 str.split() 自动处理连续空格，无需手动过滤。

    Args:
        text: 输入字符串，可为 None。

    Returns:
        单词数量；空串或 None 返回 0。

    Examples:
        >>> count_words("hello world")
        2
        >>> count_words("  a  b  c  ")
        3
    """
    if not text:
        return 0
    return len(text.split())


def reverse_words(text: str | None) -> str:
    """按单词反转字符串。

    优化：split() + 切片 [::-1] + join()，避免循环拼接。

    Args:
        text: 输入字符串，可为 None。

    Returns:
        单词反转后的字符串。

    Examples:
        >>> reverse_words("hello world")
        'world hello'
    """
    if not text:
        return ""
    return " ".join(text.split()[::-1])


def is_palindrome(text: str | None) -> bool:
    """检查字符串是否为回文（忽略空格和大小写）。

    优化：用列表推导 + join 一次性构建 cleaned 字符串，
    再用切片 [::-1] 做整体比较，替代逐字符循环。

    Args:
        text: 输入字符串，可为 None。

    Returns:
        True 如果是回文，False 否则。

    Examples:
        >>> is_palindrome("A man a plan a canal Panama")
        True
        >>> is_palindrome("hello")
        False
    """
    if text is None:
        return False
    cleaned = "".join(ch.lower() for ch in text if ch != " ")
    if not cleaned:
        return False
    return cleaned == cleaned[::-1]


def remove_duplicate_chars(text: str | None) -> str:
    """去除重复字符，保留首次出现的顺序。

    优化：用 set 记录已见字符，O(1) 查找替代 O(n) 扫描。
    用列表收集结果后 join，避免字符串拼接的 O(n²)。

    Args:
        text: 输入字符串，可为 None。

    Returns:
        去重后的字符串。

    Examples:
        >>> remove_duplicate_chars("aabbccdd")
        'abcd'
        >>> remove_duplicate_chars("abracadabra")
        'abrcd'
    """
    if not text:
        return ""
    seen: set[str] = set()
    result: list[str] = []
    for ch in text:
        if ch not in seen:
            seen.add(ch)
            result.append(ch)
    return "".join(result)
