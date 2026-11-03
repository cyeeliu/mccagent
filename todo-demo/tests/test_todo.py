"""TodoManager 单元测试"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from todo import TodoManager


def test_add():
    mgr = TodoManager()
    tid = mgr.add("学习Python")
    assert tid == 1
    assert len(mgr.list_all()) == 1
    print("✅ test_add 通过")


def test_complete():
    mgr = TodoManager()
    tid = mgr.add("写代码")
    assert mgr.complete(tid) is True
    assert len(mgr.pending()) == 0
    print("✅ test_complete 通过")


def test_complete_nonexistent():
    mgr = TodoManager()
    assert mgr.complete(999) is False
    print("✅ test_complete_nonexistent 通过")


def test_pending():
    mgr = TodoManager()
    mgr.add("任务1")
    mgr.add("任务2")
    mgr.complete(1)
    assert len(mgr.pending()) == 1
    print("✅ test_pending 通过")


def test_clear_done():
    mgr = TodoManager()
    mgr.add("任务1")
    mgr.add("任务2")
    mgr.add("任务3")
    mgr.complete(1)
    mgr.complete(3)
    cleared = mgr.clear_done()
    assert cleared == 2
    assert len(mgr.list_all()) == 1
    print("✅ test_clear_done 通过")


if __name__ == "__main__":
    test_add()
    test_complete()
    test_complete_nonexistent()
    test_pending()
    test_clear_done()
    print("\n🎉 所有测试通过！")
