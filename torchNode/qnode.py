import torch
from math import pi


class QuantumNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwargs, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.func = kwargs['func']
        ctx.grad_func = kwargs['grad']
        ctx.save_for_backward(input)

        return torch.tensor(ctx.func(input),requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        inputs = ctx.saved_tensors
        input_tensor = inputs[0]
        grad_func = ctx.grad_func
        if grad_func == 'numerical':
            grad = []
            for i in range(len(input_tensor)):
                tmp_tensor = input_tensor.clone()
                tmp_tensor[i] += 1e-5
                first_run = ctx.func(tmp_tensor)
                tmp_tensor[i] -= 2*1e-5
                second_run = ctx.func(tmp_tensor)
                grad.append((first_run-second_run)/(2*1e-5))

            result = torch.matmul(torch.vstack(grad), grad_output.t())
            return None, result.real
        grad = grad_func(input_tensor)
        result = grad* grad_output
        #print('result', result)
        return None, result

def QNode(kwargs):
    model = QuantumNode.apply
    return lambda input: model(kwargs, input)