from collections import OrderedDict


class VContainer:
    ERROR_MESS1 = "The element type is not same as the former."

    def __init__(self):
        self.container = OrderedDict()
        self.keys = []
        self.base = None

    def flash(self, key: str, element):
        if key not in self.container.keys():
            self.keys.append(key)
            self.container[key] = []
            self.container[key].append(element)
        else:
            assert type(self.container[key][0]) == type(element), self.ERROR_MESS1
            self.container[key].append(element)

    def __getitem__(self, item):
        return self.container[item]
