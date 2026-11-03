"""简单的 Todo 任务管理器"""


class TodoManager:
    def __init__(self):
        self.todos = []

    def add(self, task: str) -> int:
        """添加任务，返回任务ID"""
        todo_id = len(self.todos) + 1
        self.todos.append({"id": todo_id, "task": task, "done": False})
        return todo_id

    def complete(self, todo_id: int) -> bool:
        """标记任务为完成，返回是否成功"""
        for todo in self.todos:
            if todo["id"] == todo_id:
                todo["done"] = True
                return True
        return False

    def list_all(self) -> list:
        """列出所有任务"""
        return self.todos

    def pending(self) -> list:
        """列出未完成的任务"""
        return [t for t in self.todos if not t["done"]]

    def clear_done(self) -> int:
        """清除已完成的任务，返回清除数量"""
        before = len(self.todos)
        self.todos = [t for t in self.todos if not t["done"]]
        return before - len(self.todos)
