"""
Jacobians of neural nets: d(outputs)/d(inputs) for any chosen outputs and inputs,
which may be a model's input, its parameters, or intermediate activations.

    jacobian(outputs, inputs)          the n x d Jacobian of flattened, concatenated tensors
    capture(model, names)              record named submodules' outputs for use as either
    select_params(model, names)        parameters by dotted name or prefix
    input_jacobian(model, x, ...)      J_{y<-x}
    parameter_jacobian(model, x, ...)  J_{y<-W}
"""

import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Sequence, Union

TensorOrList = Union[Tensor, Sequence[Tensor]]


def _as_list(t: TensorOrList) -> List[Tensor]:
    return [t] if isinstance(t, Tensor) else list(t)


def _flatten_cat(tensors: Sequence[Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def jacobian(outputs: TensorOrList, inputs: TensorOrList,
             create_graph: bool = False) -> Tensor:
    """The n x d Jacobian d(outputs)/d(inputs).

    `outputs` and `inputs` are tensors, or lists of tensors, from one autograd graph;
    they are flattened and concatenated in order. Inputs need not be leaves. Computed
    in one batched backward pass, falling back to one pass per row for ops that do
    not support it.
    """
    outs = _as_list(outputs)
    ins = _as_list(inputs)
    y = _flatten_cat(outs)
    n = y.numel()
    if n == 0:
        return y.new_zeros(0, sum(i.numel() for i in ins))
    eye = torch.eye(n, dtype=y.dtype, device=y.device)
    try:
        grads = torch.autograd.grad(
            y, ins, grad_outputs=eye, is_grads_batched=True,
            retain_graph=True, create_graph=create_graph, allow_unused=True,
        )
    except RuntimeError:
        grads = _jacobian_rowwise(y, ins, n, create_graph)
    cols = []
    for g, inp in zip(grads, ins):
        if g is None:
            g = y.new_zeros(n, inp.numel())
        else:
            g = g.reshape(n, -1)
        cols.append(g)
    return torch.cat(cols, dim=1)


def _jacobian_rowwise(y: Tensor, ins: List[Tensor], n: int,
                       create_graph: bool) -> List[Optional[Tensor]]:
    """One backward pass per output row, stacked as the batched pass would be."""
    rows: List[List[Optional[Tensor]]] = [[] for _ in ins]
    for i in range(n):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1.0
        grads = torch.autograd.grad(
            y, ins, grad_outputs=grad_outputs,
            retain_graph=True, create_graph=create_graph, allow_unused=True,
        )
        for k, g in enumerate(grads):
            rows[k].append(g)
    out = []
    for k, inp in enumerate(ins):
        if all(g is None for g in rows[k]):
            out.append(None)
        else:
            filled = [g if g is not None else torch.zeros_like(inp) for g in rows[k]]
            out.append(torch.stack(filled, dim=0))
    return out


class capture:
    """Context manager recording the forward outputs of named submodules, still
    attached to the autograd graph, keyed by dotted name:

        with capture(model, ["layer2", "layer4"]) as cap:
            model(x)
        J = jacobian(cap["layer4"], cap["layer2"])
    """

    def __init__(self, model: nn.Module, names: Sequence[str]):
        self.model = model
        self.names = list(dict.fromkeys(names))
        self.acts: Dict[str, Tensor] = {}
        self._handles = []

    def __getitem__(self, name: str) -> Tensor:
        return self.acts[name]

    def __enter__(self) -> "capture":
        modules = dict(self.model.named_modules())
        for name in self.names:
            if name not in modules:
                raise KeyError(
                    f"no submodule named {name!r}; "
                    f"available: {list(modules)}"
                )

            def hook(_module, _inp, out, name=name):
                self.acts[name] = out

            self._handles.append(modules[name].register_forward_hook(hook))
        return self

    def __exit__(self, *exc) -> bool:
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def select_params(model: nn.Module, names: Sequence[str]) -> List[nn.Parameter]:
    """Parameters by dotted name, e.g. "layer3.weight", or by prefix, e.g. "layer3",
    in the order of `names`."""
    params = dict(model.named_parameters())
    out: List[nn.Parameter] = []
    for name in names:
        if name in params:
            out.append(params[name])
            continue
        prefix = name + "."
        matched = [p for k, p in params.items() if k.startswith(prefix)]
        if not matched:
            raise KeyError(f"no parameter matches {name!r}; available: {list(params)}")
        out.extend(matched)
    return out


def input_jacobian(model: nn.Module, x: Tensor,
                    output_names: Optional[Sequence[str]] = None,
                    input_names: Optional[Sequence[str]] = None,
                    **forward_kwargs) -> Tensor:
    """The input Jacobian J_{y<-x}. output_names and input_names are submodules whose
    outputs serve as the system's outputs and inputs; None means the model's output
    and its input x."""
    x = x.detach().clone().requires_grad_(True)
    watch = list(dict.fromkeys((output_names or []) + (input_names or [])))
    with capture(model, watch) as cap:
        y = model(x, **forward_kwargs)
    outs = [cap[n] for n in output_names] if output_names else [y]
    ins = [cap[n] for n in input_names] if input_names else [x]
    return jacobian(outs, ins)


def parameter_jacobian(model: nn.Module, x: Tensor,
                        output_names: Optional[Sequence[str]] = None,
                        param_names: Optional[Sequence[str]] = None,
                        **forward_kwargs) -> Tensor:
    """The parameter Jacobian J_{y<-W}. output_names as in input_jacobian; param_names
    as in select_params, None meaning every parameter."""
    with capture(model, output_names or []) as cap:
        y = model(x, **forward_kwargs)
    outs = [cap[n] for n in output_names] if output_names else [y]
    params = select_params(model, param_names) if param_names else list(model.parameters())
    return jacobian(outs, params)
