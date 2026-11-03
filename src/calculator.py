"""计算器模块 - 提供四则运算功能"""


def _validate_number(value, name="参数"):
    """校验参数是否为数字类型（int 或 float）"""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name}必须是数字类型，得到的是 {type(value).__name__}")
    return float(value)


def add(a, b):
    """加法运算: a + b

    Args:
        a: 第一个操作数 (int/float)
        b: 第二个操作数 (int/float)

    Returns:
        float: 两数之和

    Raises:
        TypeError: 参数不是数字类型时
    """
    a = _validate_number(a, "a")
    b = _validate_number(b, "b")
    return a + b


def subtract(a, b):
    """减法运算: a - b

    Args:
        a: 被减数 (int/float)
        b: 减数 (int/float)

    Returns:
        float: 两数之差

    Raises:
        TypeError: 参数不是数字类型时
    """
    a = _validate_number(a, "a")
    b = _validate_number(b, "b")
    return a - b


def multiply(a, b):
    """乘法运算: a * b

    Args:
        a: 第一个操作数 (int/float)
        b: 第二个操作数 (int/float)

    Returns:
        float: 两数之积

    Raises:
        TypeError: 参数不是数字类型时
    """
    a = _validate_number(a, "a")
    b = _validate_number(b, "b")
    return a * b


def divide(a, b):
    """除法运算: a / b

    Args:
        a: 被除数 (int/float)
        b: 除数 (int/float)

    Returns:
        float: 两数之商

    Raises:
        TypeError: 参数不是数字类型时
        ValueError: 除数为零时
    """
    a = _validate_number(a, "a")
    b = _validate_number(b, "b")
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b
