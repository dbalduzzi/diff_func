# Differential Functionalism: Jacobians Shape the Structure of Phenomenal Experience

Code for the paper by David Balduzzi, arXiv:[ID].

The paper proposes that the first-order structure of physical interactions, Jacobians, shapes the structure of experience, and introduces two measures of that structure: **transparency**, the effective rank of a Jacobian, and **integrity**, its normalised algebraic connectivity. This repository computes the measures and reproduces every figure and quoted number.

## Contents

| File | |
|---|---|
| `jacobian_measures.py` | Transparency, integrity, integral rank, Fiedler vectors, coherency and distinct ideas, closed forms for convolution, and a spanning-tree measure for comparison. Takes torch tensors or numpy arrays. |
| `jacobians.py` | Input and parameter Jacobians of any chosen neurons in a PyTorch model (`jacobian`, `capture`, `input_jacobian`, `parameter_jacobian`). |
| `diff_func.ipynb` | Works through the paper's examples section by section, with figures. |

## Running

```
pip install -r requirements.txt
jupyter notebook diff_func.ipynb
```

The notebook runs in about two minutes on a laptop CPU and writes its figures to `notebook_figs/`.

## Citation

```
@misc{balduzzi2026differential,
  author = {David Balduzzi},
  title  = {Differential Functionalism: Jacobians Shape the Structure of Phenomenal Experience},
  year   = {2026},
  note   = {arXiv:[ID]}
}
```

## License

MIT; see `LICENSE`.
