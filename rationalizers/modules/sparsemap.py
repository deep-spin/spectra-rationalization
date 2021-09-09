""""
LP-SparseMAP different strategies for rationale extraction
"""

import torch
from lpsmap import (
    Budget,
    Pair,
    Sequence,
    SequenceBudget,
    TorchFactorGraph,
    Xor,
    AtMostOne,
)


def seq_budget_smap(
    unary_scores,
    transition_scores,
    max_iter=1,
    step_size=0,
    init=True,
    budget=5,
    temperature=0.01,
):
    """
    H:SeqBudget strategy for highlights extraction
    """
    unary_scores.shape[0]

    fg = TorchFactorGraph()
    u = fg.variable_from(unary_scores)
    fg.add(SequenceBudget(u, transition_scores, budget))
    fg.solve(max_iter=max_iter, step_size=step_size)
    u.value.cuda()
    return u.value[:, 0]


def matching_smap(scores, max_iter=5, temperature=1, init=True, budget=None):
    """
    M:XORAtMostOne strategy for matchings extraction
    """

    m, n = scores.shape
    fg = TorchFactorGraph()
    z = fg.variable_from(scores / temperature)
    for i in range(m):
        fg.add(Xor(z[i, :]))
    for j in range(n):
        fg.add(AtMostOne(z[:, j]))  # some cols may be 0
    fg.lp_map_solve(max_iter=max_iter)
    return z.value.cuda()


def matching_smap_atmostone(scores, max_iter=5, temperature=1, init=True, budget=None):
    """
    M:AtMostOne2 strategy for matchings extraction
    """

    m, n = scores.shape
    fg = TorchFactorGraph()
    z = fg.variable_from(scores / temperature)
    for i in range(m):
        fg.add(AtMostOne(z[i, :]))
    for j in range(n):
        fg.add(AtMostOne(z[:, j]))  # some cols may be 0
    fg.solve(max_iter=max_iter)
    return z.value.cuda()


def matching_smap_atmostone_budget(
    scores, max_iter=5, temperature=1, init=True, budget=None
):
    """
    M:Budget strategy for matchings extraction
    """

    m, n = scores.shape
    fg = TorchFactorGraph()
    z = fg.variable_from(scores / temperature)
    fg.add(Budget(z, budget=budget))
    for i in range(m):
        fg.add(AtMostOne(z[i, :]))
    for j in range(n):
        fg.add(AtMostOne(z[:, j]))  # some cols may be 0
    fg.solve(max_iter=max_iter)
    return z.value.cuda()
