from spaghettini import quick_register


@quick_register
class TransformerSchedulerWrapper:
    def __init__(self, get_fn, **kwargs):
        self.get_fn = get_fn
        self.kwargs = kwargs

    def get_scheduler(self, optimizer):
        return self.get_fn(optimizer, **self.kwargs)
