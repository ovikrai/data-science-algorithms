###############################################
# Structure: Stack
# FIFO: First In First Out
# Complexity:
#   Read:
#   Write:
###############################################
class Stack(object):
    def __init__(self, elements=None):
        if elements is None:
            elements = list()
        self.elements = elements

    def size(self):
        return len(self.elements) - 1

    def is_empty(self):
        if self.size() == 0:
            return True
        else:
            return False

    # Remove from top
    def pop(self):
        self.elements.pop(self.size() - 1)

    # Add to top
    def push(self, element):
        self.elements.append(element)
