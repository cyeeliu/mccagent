"""一个简单的 Python 示例文件"""


def greet(name: str) -> str:
    """返回问候语"""
    return f"你好，{name}！"


def main() -> None:
    print(greet("世界"))
    print(greet("Python"))


if __name__ == "__main__":
    main()
