from typing import List, NoReturn


class Matrix:
    matrix: List[List[float]]
    column_size: int

    # Matrix initialization
    def __int__(self, n: int, s_type='float'):
        self.column_size = n

        # INITIALIZATION PROCEDURE
        for i in range(0, n + 1, 1):
            # Initialize with function provided values
            self.matrix.append([])
            for j in range(0, n + 1, 1):
                # Matrix initialization with Zeros
                self.matrix[i].append(0.0)

    def size(self) -> int:
        return len(matrix) * self.column_size

    def update_element(self, value, i, j):
        self.matrix[i][j] = value

    def get_element(self, i, j):
        return self.matrix[i][j]

    def get_object_original(self):
        return self.matrix

    def render(self) -> NoReturn:
        print('########## START: MATRIX  RENDERING REPRESENTATION #########')
        print('########## | MATRIX: SIZE', self.size())
        print('########## |', self.matrix)
        print('########## END: MATRIX RENDERING REPRESENTATION ######### \n')
