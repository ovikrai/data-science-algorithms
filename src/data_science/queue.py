###############################################
# Structure: Queue
# LIFO: Last In, First Out
# Complexity:
#   Read:
#   Write:
###############################################
class Queue(object):
    elements: list

    def __init__(self, elements=list()):
        self.elements = elements

    def size(self):
        return len(self.elements) - 1

    def is_empty(self):
        if self.size() == 0:
            return True
        else:
            return False

    # Add elements
    def enqueue(self, element):
        self.elements.append(element)

    # Remove elements
    def dequeue(self):
        self.elements.pop(0)

    def render(self) -> NoReturn:
        print('########## START: QUEUE RENDERING REPRESENTATION #########')
        print('########## | QUEUE: SIZE', self.size())
        print('########## |', self.elements)
        print('########## END: QUEUE RENDERING REPRESENTATION ######### \n')
