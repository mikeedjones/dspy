import ujson

from dspy.primitives.module import BaseModule

# NOTE: Note: It's important (temporary decision) to maintain named_parameters that's different in behavior from
# named_sub_modules for the time being.


class ServingModule(BaseModule):
    def __init__(self):
        pass

    async def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_state(self, state):
        for name, param in self.named_parameters():
            param.load_state(state[name])

    def load(self, path):
        with open(path) as f:
            self.load_state(ujson.loads(f.read()))
