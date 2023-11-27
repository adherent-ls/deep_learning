from base.base.base import Base


class BaseDictCall(Base):
    def set(self, ink='images', ouk=None):
        self.ink = ink
        if ouk is not None:
            self.ouk = ouk
        else:
            self.ouk = ink

    def __call__(self, data):
        if len(self.ink) != 0:
            if isinstance(self.ink, str):
                ink = [self.ink]
            else:
                ink = self.ink
            xs = []
            for item in ink:
                xs.append(data[item])
        else:
            xs = []

        y = self.forward(*xs)

        if isinstance(self.ouk, list) or isinstance(self.ouk, tuple):
            for key, data_item in zip(self.ouk, y):
                data[key] = data_item
        else:
            data[self.ouk] = y
        return data

    def forward(self, *args):
        raise NotImplemented
