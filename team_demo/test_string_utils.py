"""字符串工具库单元测试。

测试内容：
  1. 功能正确性测试（原始版本 + 优化版本）
  2. 两版本结果一致性测试
  3. 性能对比测试
"""

import time

import pytest

from string_utils import (
    count_words as count_words_v1,
    reverse_words as reverse_words_v1,
    is_palindrome as is_palindrome_v1,
    remove_duplicate_chars as remove_dup_v1,
)
from string_utils_optimized import (
    count_words as count_words_v2,
    reverse_words as reverse_words_v2,
    is_palindrome as is_palindrome_v2,
    remove_duplicate_chars as remove_dup_v2,
)


# ============================================================
# 1. 功能正确性测试
# ============================================================

class TestCountWords:
    """测试单词计数功能。"""

    @pytest.mark.parametrize("text,expected", [
        ("hello world", 2),
        ("one two three four", 4),
        ("", 0),
        ("single", 1),
        ("  multiple   spaces  between  ", 3),
        ("a b c d e f", 6),
    ])
    def test_v1(self, text, expected):
        assert count_words_v1(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("hello world", 2),
        ("one two three four", 4),
        ("", 0),
        ("single", 1),
        ("  multiple   spaces  between  ", 3),
        ("a b c d e f", 6),
    ])
    def test_v2(self, text, expected):
        assert count_words_v2(text) == expected

    def test_none_input(self):
        assert count_words_v1(None) == 0
        assert count_words_v2(None) == 0


class TestReverseWords:
    """测试单词反转功能。"""

    @pytest.mark.parametrize("text,expected", [
        ("hello world", "world hello"),
        ("a b c", "c b a"),
        ("", ""),
        ("single", "single"),
        ("one two three", "three two one"),
    ])
    def test_v1(self, text, expected):
        assert reverse_words_v1(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("hello world", "world hello"),
        ("a b c", "c b a"),
        ("", ""),
        ("single", "single"),
        ("one two three", "three two one"),
    ])
    def test_v2(self, text, expected):
        assert reverse_words_v2(text) == expected

    def test_none_input(self):
        assert reverse_words_v1(None) == ""
        assert reverse_words_v2(None) == ""


class TestIsPalindrome:
    """测试回文检查功能。"""

    @pytest.mark.parametrize("text", [
        "A man a plan a canal Panama",
        "racecar",
        "Was it a car or a cat I saw",
        "noon",
        "a",
        "abba",
    ])
    def test_v1_true(self, text):
        assert is_palindrome_v1(text) is True

    @pytest.mark.parametrize("text", [
        "A man a plan a canal Panama",
        "racecar",
        "Was it a car or a cat I saw",
        "noon",
        "a",
        "abba",
    ])
    def test_v2_true(self, text):
        assert is_palindrome_v2(text) is True

    @pytest.mark.parametrize("text", [
        "hello",
        "world",
        "abc",
        "not a palindrome",
    ])
    def test_v1_false(self, text):
        assert is_palindrome_v1(text) is False

    @pytest.mark.parametrize("text", [
        "hello",
        "world",
        "abc",
        "not a palindrome",
    ])
    def test_v2_false(self, text):
        assert is_palindrome_v2(text) is False

    def test_edge_cases(self):
        assert is_palindrome_v1(None) is False
        assert is_palindrome_v2(None) is False
        assert is_palindrome_v1("") is False
        assert is_palindrome_v2("") is False
        assert is_palindrome_v1("   ") is False
        assert is_palindrome_v2("   ") is False


class TestRemoveDuplicateChars:
    """测试去重功能。"""

    @pytest.mark.parametrize("text,expected", [
        ("aabbccdd", "abcd"),
        ("abracadabra", "abrcd"),
        ("", ""),
        ("abcdef", "abcdef"),
        ("aaaaaa", "a"),
        ("abcabc", "abc"),
    ])
    def test_v1(self, text, expected):
        assert remove_dup_v1(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("aabbccdd", "abcd"),
        ("abracadabra", "abrcd"),
        ("", ""),
        ("abcdef", "abcdef"),
        ("aaaaaa", "a"),
        ("abcabc", "abc"),
    ])
    def test_v2(self, text, expected):
        assert remove_dup_v2(text) == expected

    def test_none_input(self):
        assert remove_dup_v1(None) == ""
        assert remove_dup_v2(None) == ""


# ============================================================
# 2. 两版本结果一致性测试
# ============================================================

class TestConsistency:
    """验证原始版本和优化版本结果完全一致。"""

    TEST_STRINGS = [
        "hello world",
        "A man a plan a canal Panama",
        "aabbccdd",
        "abracadabra",
        "  multiple   spaces  ",
        "the quick brown fox jumps over the lazy dog",
        "racecar",
        "",
        "x" * 100,
        "ab " * 50,
    ]

    @pytest.mark.parametrize("text", TEST_STRINGS)
    def test_count_words(self, text):
        assert count_words_v1(text) == count_words_v2(text)

    @pytest.mark.parametrize("text", TEST_STRINGS)
    def test_reverse_words(self, text):
        assert reverse_words_v1(text) == reverse_words_v2(text)

    @pytest.mark.parametrize("text", TEST_STRINGS)
    def test_is_palindrome(self, text):
        assert is_palindrome_v1(text) == is_palindrome_v2(text)

    @pytest.mark.parametrize("text", TEST_STRINGS)
    def test_remove_dup(self, text):
        assert remove_dup_v1(text) == remove_dup_v2(text)


# ============================================================
# 3. 性能对比测试
# ============================================================

class TestPerformance:
    """对比原始版本与优化版本的性能。"""

    # 构造大字符串用于性能测试
    BIG_TEXT = " ".join(["word"] * 5000)
    BIG_DUP = "abcdefghij" * 1000
    BIG_PALIN = "a" * 5000 + "b" + "a" * 5000

    def _time_it(self, func, *args, runs=5):
        """多次运行取平均耗时。"""
        total = 0.0
        for _ in range(runs):
            start = time.perf_counter()
            func(*args)
            total += time.perf_counter() - start
        return total / runs

    def test_count_words_v2_faster(self):
        t1 = self._time_it(count_words_v1, self.BIG_TEXT)
        t2 = self._time_it(count_words_v2, self.BIG_TEXT)
        print(f"\n  count_words: v1={t1:.6f}s  v2={t2:.6f}s  speedup={t1/t2:.1f}x")
        assert t2 <= t1

    def test_reverse_words_v2_faster(self):
        t1 = self._time_it(reverse_words_v1, self.BIG_TEXT)
        t2 = self._time_it(reverse_words_v2, self.BIG_TEXT)
        print(f"\n  reverse_words: v1={t1:.6f}s  v2={t2:.6f}s  speedup={t1/t2:.1f}x")
        assert t2 <= t1

    def test_is_palindrome_v2_faster(self):
        t1 = self._time_it(is_palindrome_v1, self.BIG_PALIN)
        t2 = self._time_it(is_palindrome_v2, self.BIG_PALIN)
        print(f"\n  is_palindrome: v1={t1:.6f}s  v2={t2:.6f}s  speedup={t1/t2:.1f}x")
        assert t2 <= t1

    def test_remove_dup_v2_faster(self):
        t1 = self._time_it(remove_dup_v1, self.BIG_DUP)
        t2 = self._time_it(remove_dup_v2, self.BIG_DUP)
        print(f"\n  remove_dup: v1={t1:.6f}s  v2={t2:.6f}s  speedup={t1/t2:.1f}x")
        assert t2 <= t1
