###############################################
# Structure: Bag, abstraction of the mathematical set
# Complexity:
#   Read:
#   Write:
###############################################
class Bag(object):
    elements: set

    def __init__(self):
        self.elements = set()


    def size(self) -> int:
        return len(self.elements)

    def is_empty(self):
        if self.size() == 0:
            return True
        else:
            return False

    def get_element(self, target_element):
        # SEARCH ELEMENT
        for element in self.elements:
            if target_element == element:
                return element
            else:
                return -1

    def add_element(self, element):
        self.elements.add(element)

    def remove_element(self, element):
        self.elements.remove(element)

    def render(self):
        print('########## START: BAG RENDERING REPRESENTATION #########')
        print('########## | BAG: SIZE', self.size())
        print('########## |', self.elements)
        print('########## END: BAG RENDERING REPRESENTATION ######### \n')

