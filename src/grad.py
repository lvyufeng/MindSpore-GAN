from mindspore._c_expression import GradOperation_
from mindspore.common.api import ms_function
from mindspore.ops import stop_gradient

grad_func = GradOperation_('grad', True, False, False, False)
grad_cell = GradOperation_('grad', False, True, False, False)
def value_and_grad(fn, params=None, has_aux=False):
    if params is None:
        grad_ = grad_func
    else:
        grad_ = grad_cell

    @ms_function
    def fn_aux(*args):
        outputs = fn(*args)
        no_grad_outputs = ()
        for out in outputs[1:]:
            no_grad_outputs += (stop_gradient(out),)
        return outputs[0], no_grad_outputs

    if has_aux:
        fn_ = fn_aux
    else:
        fn_ = fn

    @ms_function
    def value_and_grad_f(*args):
        values = fn_(*args)
        if params is None:
            grads = grad_(fn_)(*args)
        else:
            grads = grad_(fn_, params)(*args)
        return values, grads
    return value_and_grad_f

def grad(fn, params=None, has_aux=False):
    value_and_grad_f = value_and_grad(fn, params, has_aux)
    def grad_f(*args):
        _, g = value_and_grad_f(*args)
        return g
    return grad_f