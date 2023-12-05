from utils import swim, sink, swap


# Binary Heap Priority Queue
class PriorityQueue:
    # priority queue
    pq: list
    n: int

    def __init__(self, max_n):
        max_n = max_n + 1
        self.pq = [None] * max_n
        self.n = 0

    def is_empty(self) -> bool:
        return self.n == 0

    def size(self) -> int:
        return self.n

    def insert(self, i: int, value):
        self.pq[i] = value
        swim(self.pq, self.n + 1)

    def delete(self, i):
        item = self.pq[i]
        swap(self.pq, i, self.n - 1)
        swim(self.pq, i)
        sink(self.pq, i)
        self.pq[i] = None
        return item

    def contains(self, i: int):
        if self.pq[i] is None:
            return False
        else:
            return True

    def change(self, i, value):
        if not self.contains(i):
            raise Exception('Index is not in the priority queue')
        self.pq[i] = value
        swim(self.pq, i)
        sink(self.pq, i)
