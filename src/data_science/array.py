from typing import Any, NoReturn
from algos.utils import allocate, deallocate


###############################################
# Structure: Array: An immutable set of elements with order
# Complexity:
#   Read:
#   Write:
###############################################
class Array(object):
    elements: tuple

    def __init__(self, size: int):
        init_list = allocate(None, size)
        self.elements = tuple(init_list)
        deallocate(init_list)

    def size(self) -> int:
        return len(self.elements)

    def is_empty(self) -> bool:
        if self.size() == 0:
            return True
        else:
            return False

    def set_element(self, key: int, value: Any) -> NoReturn:
        update_list = list(self.elements)
        update_list[key] = value
        self.elements = tuple(update_list)
        deallocate(update_list)

    def get_element(self, key: int) -> Any:
        return self.elements[key]

    def render(self) -> NoReturn:
        print('########## START: ARRAY RENDERING REPRESENTATION #########')
        print('########## | ARRAY: SIZE', self.size())
        print('########## |', self.elements)
        print('########## END: ARRAY RENDERING REPRESENTATION ######### \n')
