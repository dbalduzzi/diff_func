"""
Measures of Jacobian structure, from "Differential Functionalism: Jacobians Shape
the Structure of Phenomenal Experience".

    transparency(A)            effective rank (sum s^2)^2 / sum s^4
    integrity(A)               1 - sigma_2(D_r^-1/2 |A| D_c^-1/2)
    integral_rank(A)           transparency * integrity
    fiedler(A)                 Fiedler vectors of G(A): the weakest seam
    coherency, distinct_ideas  coherency matrix J J^T and its blocks
    conv_*                     closed forms for circular convolution
    effective_rank_*, entropic_rank   alternative forms of effective rank
    spanning_tree_cohesion     normalised spanning-tree count of G(A) (arXiv:2609.09306)

G(A) is the bipartite graph with edge weights |A_ij|. The measures take torch
tensors (returning 0-dim tensors) or anything numpy can read (returning floats).
"""

import math

import numpy as _np
import torch
from torch import Tensor


# --- helpers ------------------------------------------------------------------

def _as_tensor(A):
    if isinstance(A, Tensor):
        return A, True
    return torch.as_tensor(_np.asarray(A, dtype=_np.float64)), False


def _out(x, was_tensor):
    if was_tensor:
        return x if isinstance(x, Tensor) else torch.tensor(x, dtype=torch.float64)
    return float(x)


def _flexible(fn):
    """Let a Tensor -> 0-dim Tensor function also take numpy input."""
    def wrapped(A, *args, **kwargs):
        T, was_tensor = _as_tensor(A)
        return _out(fn(T, *args, **kwargs), was_tensor)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


def _live(absA: Tensor):
    return absA.sum(1) > 0, absA.sum(0) > 0


def drop_inert(A: Tensor) -> Tensor:
    """A without its identically zero rows and columns."""
    rows, cols = _live(A.abs())
    return A[rows][:, cols]


def _nonzero_singular_values(A: Tensor) -> Tensor:
    s = torch.linalg.svdvals(A)
    tol = max(A.shape) * torch.finfo(s.dtype).eps * s.max() if s.numel() > 0 else 0.0
    return s[s > tol]


# --- transparency ---------------------------------------------------------------

def effective_rank(A: Tensor) -> Tensor:
    """(sum s_i^2)^2 / sum s_i^4 over the nonzero singular values; 0 if A == 0.
    In [1, rank(A)]: 1 for rank one, rank(A) when the singular values are equal."""
    s = _nonzero_singular_values(A)
    if s.numel() == 0:
        return A.new_zeros(())
    ss = s.square()
    return ss.sum().square() / ss.square().sum()


transparency = _flexible(effective_rank)
transparency.__name__ = "transparency"


def effective_rank_trace(A: Tensor) -> Tensor:
    """Effective rank as tr(G)^2 / tr(G^2), G the smaller Gram matrix of A."""
    G = A @ A.T if A.shape[0] <= A.shape[1] else A.T @ A
    tr = torch.trace(G)
    if tr == 0:
        return A.new_zeros(())
    return tr.square() / torch.trace(G @ G)


def effective_rank_collision(A: Tensor) -> Tensor:
    """Effective rank as 1 / sum p_i^2, with p_i = s_i^2 / sum_j s_j^2."""
    s = _nonzero_singular_values(A)
    if s.numel() == 0:
        return A.new_zeros(())
    p = s.square() / s.square().sum()
    return 1.0 / p.square().sum()


def effective_rank_cv(A: Tensor) -> Tensor:
    """Effective rank as r / (1 + cv^2), cv the coefficient of variation of the s_i^2."""
    s = _nonzero_singular_values(A)
    if s.numel() == 0:
        return A.new_zeros(())
    x = s.square()
    return s.numel() / (1 + x.var(unbiased=False) / x.mean().square())


def entropic_rank(A):
    """exp(H(q)), q_i = s_i / sum_j s_j (Roy & Vetterli 2007); 0 if A == 0."""
    T, was_tensor = _as_tensor(A)
    s = _nonzero_singular_values(T)
    if s.numel() == 0:
        return _out(T.new_zeros(()), was_tensor)
    q = s / s.sum()
    return _out(torch.exp(-(q * q.log()).sum()), was_tensor)


# --- integrity ------------------------------------------------------------------

def normalised_adjacency(A, drop: bool = True):
    """D_r^-1/2 |A| D_c^-1/2 on the live rows and columns (all of them if not drop).
    Returns (matrix, row mask, column mask)."""
    T, _ = _as_tensor(A)
    absA = T.abs()
    rows, cols = _live(absA)
    if not drop:
        rows, cols = torch.ones_like(rows), torch.ones_like(cols)
    B = absA[rows][:, cols]
    if B.numel() == 0:
        return B, rows, cols
    tiny = torch.finfo(B.dtype).tiny
    r, c = B.sum(1).clamp_min(tiny), B.sum(0).clamp_min(tiny)
    return B / r.sqrt()[:, None] / c.sqrt()[None, :], rows, cols


def _integrity(T: Tensor, drop: bool = True) -> Tensor:
    absA = T.abs()
    rows, cols = _live(absA)
    if not bool(rows.any()):
        return T.new_zeros(())
    if not drop and not (bool(rows.all()) and bool(cols.all())):
        return T.new_zeros(())                      # an inert node is isolated
    B = absA[rows][:, cols]
    if min(B.shape) == 1:
        return T.new_ones(())                       # sigma_2 := 0
    M, _, _ = normalised_adjacency(B)
    return (1 - torch.linalg.svdvals(M)[1]).clamp(0.0, 1.0)


def integrity(A, drop: bool = True):
    """1 - sigma_2(D_r^-1/2 |A| D_c^-1/2): the normalised algebraic connectivity of G(A).

    In [0, 1]; 0 iff G(A) is disconnected or A == 0; 1 iff |A| has rank one, or a
    single row or column. Inert rows and columns are removed first; with drop=False
    they count as isolated nodes, giving 0.
    """
    T, was_tensor = _as_tensor(A)
    return _out(_integrity(T, drop), was_tensor)


def integral_rank(A, drop: bool = True):
    """transparency(A) * integrity(A); in [0, rank(A)]."""
    T, was_tensor = _as_tensor(A)
    return _out(effective_rank(T) * _integrity(T, drop), was_tensor)


def fiedler(A):
    """Fiedler vectors (u, v) of G(A) for rows and columns, as numpy arrays. Signs give
    the two sides of the weakest seam; inert nodes get 0."""
    T, _ = _as_tensor(A)
    M, rows, cols = normalised_adjacency(T)
    u, v = _np.zeros(T.shape[0]), _np.zeros(T.shape[1])
    if min(M.shape) < 2:
        return u, v
    U, _, Vh = torch.linalg.svd(M)
    absA = T.abs()[rows][:, cols]
    u[rows.numpy()] = (U[:, 1] / absA.sum(1).sqrt()).numpy()
    v[cols.numpy()] = (Vh[1] / absA.sum(0).sqrt()).numpy()
    return u, v


# --- distinct ideas ---------------------------------------------------------------

def coherency(J):
    """The coherency matrix J J^T."""
    T, was_tensor = _as_tensor(J)
    G = T @ T.T
    return G if was_tensor else G.numpy()


def distinct_ideas(J, tol: float = 1e-9) -> int:
    """Number of blocks into which J J^T can be permuted: connected components of the
    graph with an edge i-j when |G_ij| > tol * max |G|."""
    from scipy.sparse.csgraph import connected_components
    J = J.detach().numpy() if isinstance(J, Tensor) else _np.asarray(J, dtype=_np.float64)
    G = _np.abs(coherency(J))
    if G.max() == 0:
        return G.shape[0]
    return int(connected_components(G > tol * G.max(), directed=False)[0])


# --- circular convolution ----------------------------------------------------------

def _kernel_on_grid(k, N):
    k = _np.asarray(k, dtype=_np.float64)
    if k.shape == (N, N):
        return k                                    # already periodic, origin at [0, 0]
    a, b = k.shape[0] // 2, k.shape[1] // 2
    g = _np.zeros((N, N))
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            g[(i - a) % N, (j - b) % N] += k[i, j]
    return g


def conv_matrix(k, N: int):
    """N^2 x N^2 Jacobian of circular convolution with k on an N x N grid,
    J[(x,y),(p,q)] = k[x-p, y-q]. k is centred on its middle entry, or is N x N with
    its origin at [0, 0]."""
    g = _kernel_on_grid(k, N)
    idx = _np.arange(N)
    dx = (idx[:, None] - idx[None, :]) % N
    return g[dx[:, None, :, None], dx[None, :, None, :]].reshape(N * N, N * N)


def conv_transparency(k, N: int) -> float:
    """transparency(conv_matrix(k, N)) from the kernel's DFT."""
    p = _np.abs(_np.fft.fft2(_kernel_on_grid(k, N))) ** 2
    return 0.0 if p.sum() == 0 else float(p.sum() ** 2 / (p ** 2).sum())


def conv_integrity(k, N: int) -> float:
    """integrity(conv_matrix(k, N)) = 1 - max_{w != 0} |phi(w)|, phi the DFT of |k| / ||k||_1."""
    g = _np.abs(_kernel_on_grid(k, N))
    if g.sum() == 0:
        return 0.0
    phi = _np.abs(_np.fft.fft2(g / g.sum())).ravel()
    return float(1 - phi[1:].max())


# --- spanning trees ----------------------------------------------------------------
# log tau(G) by the matrix-tree theorem: delete one node of the smaller side, take the
# Schur complement S of the Laplacian, and read log det S from its eigenvalues, which
# stay reliable on near-disconnected graphs where a direct determinant does not.

def _logdet_spd(S: Tensor, scale: Tensor) -> Tensor:
    if S.numel() == 0:
        return S.new_zeros(())                      # a single row or column is connected
    eigs = torch.linalg.eigvalsh(S)
    tol = 10 * S.shape[-1] * torch.finfo(S.dtype).eps * scale.abs().max()
    if bool((eigs < tol).any()):
        return S.new_full((), float("-inf"))        # disconnected
    return eigs.log().sum()


def log_kirchhoff(A: Tensor) -> Tensor:
    """log of the number of spanning trees of G(A), weighted by A (nonnegative);
    -inf if G(A) is disconnected."""
    n, d = A.shape
    dr, dc = A.sum(1), A.sum(0)
    if n == 0 or d == 0 or bool((dr <= 0).any()) or bool((dc <= 0).any()):
        return A.new_full((), float("-inf"))
    if d <= n:
        B = A[:, :-1]
        S = torch.diag(dc[:-1]) - B.T @ (B / dr[:, None])
        return dr.log().sum() + _logdet_spd(S, dc[:-1])
    B = A[1:, :]
    S = torch.diag(dr[1:]) - (B / dc[None, :]) @ B.T
    return dc.log().sum() + _logdet_spd(S, dr[1:])


def _spanning_tree_cohesion(A: Tensor) -> Tensor:
    absA = A.abs()
    n, d = absA.shape
    if n == 0 or d == 0 or absA.max() == 0:
        return A.new_zeros(())
    lk = log_kirchhoff(absA / absA.mean())
    lk = lk - ((d - 1) * math.log(n) + (n - 1) * math.log(d))   # tau(K_{n,d})
    return (lk / (n + d - 1)).exp()


spanning_tree_cohesion = _flexible(_spanning_tree_cohesion)
spanning_tree_cohesion.__name__ = "spanning_tree_cohesion"
spanning_tree_cohesion.__doc__ = """(tau(G) / tau(K_{n,d}))^(1/(n+d-1)) for |A| rescaled to unit
mean: in [0, 1], 0 iff G(A) is disconnected, 1 iff all |A_ij| are equal. Same zero set as
integrity, but it averages over the whole Laplacian spectrum, so a single weak seam barely
registers."""
