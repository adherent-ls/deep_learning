from base.base.base import Base


class BaseInstanceCall(Base):
    def set(self):
        pass

    def __call__(self, *data):
        y = self.forward(*data)
        return y

    def forward(self, *args):
        raise NotImplemented
