import mindspore.nn as nn

class Generator(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)

class Critic(nn.Cell):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)