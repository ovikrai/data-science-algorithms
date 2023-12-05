from typing import List, Tuple
from algos.utils import allocate, deallocate


class Curve:
    curve: List[Tuple[float]]
    column: List
    column_size: int

    # Matrix initialization
    def __int__(self, n: int):
        self.column_size = n
        self.column = allocate(0.0, n)
        for i in range(0, n + 1, 1):
            self.curve.append(tuple(self.column))
        deallocate(self.column)

    def size(self) -> int:
        return len(self.column) * column_size

    def set_element(self, value: float, i: int, j: int):
        self.column = list(self.curve[i])
        self.column[j] = value
        self.curve[i] = tuple(self.column)

    def get_element(self, i: int, j: int):
        return self.curve[i][j]

    def get_object_original(self):
        return self.curve

    def get_object_copy(self):
        return self.curve.copy()


    def render(self) -> NoReturn:
        print('########## START: CURVE RENDERING REPRESENTATION #########')
        print('########## | CURVE: SIZE', self.size())
        print('########## |', self.curve)
        print('########## END: CURVE RENDERING REPRESENTATION ######### \n')
