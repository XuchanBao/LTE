"""
Kemeny implementation based on https://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html
"""

from spaghettini import quick_register
import itertools

import numpy as np
from scipy.stats import mode
import torch
from lp_solve import lp_solve


from src.utils.voting_utils import get_one_hot


def winner_uniqueness_given_scores(scores):
    score_diff = scores - np.max(scores, axis=1)[..., None]
    tie_counts = np.sum(np.maximum(score_diff + 0.5, 0) * 2, axis=1) - 1
    return tie_counts == 0


@quick_register
def get_plurality(one_hot=False):
    def plurality(votes, utilities=None, one_hot_repr=one_hot):
        # Don't use utilities.
        utilities = None

        # votes: (batch_size, # of voters, # of candidates)
        # ____Select the top votes of the voters. ____
        if isinstance(votes, torch.Tensor):
            top_votes = votes[:, :, 0].detach().cpu().numpy()
        else:
            top_votes = votes[:, :, 0]

        # ____ Pick the most popular candidate. ____
        winner = mode(top_votes, axis=1).mode
        winner = winner.squeeze()

        unique = []
        for batch_i in range(len(top_votes)):
            counts = np.unique(top_votes[batch_i], return_counts=True)[1]
            unique.append(sum(counts == max(counts)) == 1)

        # ____ Cast back to torch tensor, if votes was a torch tensor. ____
        if isinstance(votes, torch.Tensor):
            winner = torch.from_numpy(winner).type_as(votes)
            # winner = torch.Tensor(winner).type_as(votes)

        # ____ Optionally turn to one hot representation. ____
        num_candidates = votes.shape[2]
        winner = get_one_hot(winner, num_candidates) if one_hot_repr else winner

        return winner, np.array(unique)
    return plurality


@quick_register
def get_borda(one_hot=False):
    def borda(votes, utilities=None, one_hot_repr=one_hot):
        # Don't use utilities.
        utilities = None

        # ____ Compute borda scores for each candidate. ____

        if isinstance(votes, torch.Tensor):
            votes_np = votes.detach().cpu().numpy()
        else:
            votes_np = votes

        bs, n_voters, n_cands = votes_np.shape

        # borda_scores: (batch_size, # of candidates)
        borda_scores = np.sum((n_cands - 1 - np.argsort(votes_np, axis=2)), axis=1)

        # compute borda winner
        winner = np.argmax(borda_scores, axis=1)
        if isinstance(votes, torch.Tensor):
            winner = torch.from_numpy(winner).type_as(votes)

        winner = get_one_hot(winner, n_cands) if one_hot_repr else winner

        # check ties and compute # of unique cases in batch
        unique = winner_uniqueness_given_scores(borda_scores)

        return winner, unique

    return borda


@quick_register
def get_copeland(one_hot=False):
    def copeland(votes, utilities=None, one_hot_repr=one_hot):
        if isinstance(votes, torch.Tensor):
            votes_np = votes.detach().cpu().numpy()
        else:
            votes_np = votes

        bs, n_voters, n_cands = votes_np.shape

        # compute copeland winner
        pref_thres = n_voters * 0.5

        cand_position = np.argsort(votes_np, axis=2)

        pairwise_wins = np.zeros((bs, n_cands))
        for cand_a in range(n_cands):
            for cand_b in range(cand_a + 1, n_cands):
                num_a_win_b = np.sum(cand_position[:, :, cand_a] < cand_position[:, :, cand_b], axis=1)

                pairwise_wins[:, cand_a] += (num_a_win_b > pref_thres)
                pairwise_wins[:, cand_b] += (num_a_win_b < pref_thres)

        winner = np.argmax(pairwise_wins, axis=1)

        if isinstance(votes, torch.Tensor):
            winner = torch.from_numpy(winner).type_as(votes)
        winner = get_one_hot(winner, n_cands) if one_hot_repr else winner

        unique = winner_uniqueness_given_scores(pairwise_wins)

        return winner, unique
    return copeland


@quick_register
def get_maximin(one_hot=False):
    def maximin(votes, utilities=None, one_hot_repr=one_hot):

        if isinstance(votes, torch.Tensor):
            votes_np = votes.detach().cpu().numpy()
        else:
            votes_np = votes

        bs, n_voters, n_cands = votes_np.shape

        # compute the maximin winner

        cand_position = np.argsort(votes_np, axis=2)
        pairwise_pref = np.ones((bs, n_cands, n_cands)) * n_voters

        for cand_a in range(n_cands):
            for cand_b in range(cand_a + 1, n_cands):
                num_a_win_b = np.sum(cand_position[:, :, cand_a] < cand_position[:, :, cand_b], axis=1)

                pairwise_pref[:, cand_a, cand_b] = num_a_win_b
                pairwise_pref[:, cand_b, cand_a] = n_voters - num_a_win_b

        pairwise_min = np.min(pairwise_pref, axis=2)

        winner = np.argmax(pairwise_min, axis=1)

        if isinstance(votes, torch.Tensor):
            winner = torch.from_numpy(winner).type_as(votes)
        winner = get_one_hot(winner, n_cands) if one_hot_repr else winner

        unique = winner_uniqueness_given_scores(pairwise_min)

        return winner, unique
    return maximin


@quick_register
def get_utilitarian(one_hot=False):
    def utilitarian(votes, utilities, one_hot_repr=one_hot):
        # Don't use votes, except for getting shape
        bs, n_voters, n_cands = votes.shape

        if isinstance(utilities, torch.Tensor):
            utilities_np = utilities.detach().cpu().numpy()
        else:
            utilities_np = utilities

        # Get the total amount of utility assigned to each candidate.
        candidate_utilities = utilities_np.sum(axis=1)

        # Declare as winner the candidate that got the most utility points.
        winner = np.argmax(candidate_utilities, axis=1)

        if isinstance(utilities, torch.Tensor):
            winner = torch.from_numpy(winner).type_as(utilities)

        # Get one hot representation is asked.
        winner = get_one_hot(winner, n_cands) if one_hot_repr else winner

        return winner, np.ones((len(utilities_np), )).astype(np.bool)

    return utilitarian


@quick_register
def get_rawlsian(one_hot=False):
    # W(u_1, ..., u_n) = min(u_i)

    def rawlsian(votes, utilities, one_hot_repr=one_hot):
        # Don't use votes, except for getting shape
        bs, n_voters, n_cands = votes.shape

        if isinstance(utilities, torch.Tensor):
            utilities_np = utilities.detach().cpu().numpy()
        else:
            utilities_np = utilities

        candidate_min_utilities = utilities_np.min(axis=1)

        # Declare as winner the candidate that got the max mininum utility points.
        winner = np.argmax(candidate_min_utilities, axis=1)

        if isinstance(utilities, torch.Tensor):
            winner = torch.from_numpy(winner).type_as(utilities)

        winner = get_one_hot(winner, n_cands) if one_hot_repr else winner

        return winner, np.ones((len(utilities_np), )).astype(np.bool)
    return rawlsian


@quick_register
def get_egalitarian(one_hot=False, penalty_lambda=0.5):
    # inequality penalized
    # W(u_1, ..., u_n) = sum(u_i) - lambda * sum(u_i - min(u_i))

    def egalitarian(votes, utilities, one_hot_repr=one_hot):
        # Don't use votes, except for getting shape
        bs, n_voters, n_cands = votes.shape

        if isinstance(utilities, torch.Tensor):
            utilities_np = utilities.detach().cpu().numpy()
        else:
            utilities_np = utilities

        candidate_utilities = utilities_np.sum(axis=1)
        candidate_min_utilities = utilities_np.min(axis=1)

        candidate_penalized_utilities = (1 - penalty_lambda) * candidate_utilities \
                                        + penalty_lambda * n_voters * candidate_min_utilities

        # Declare as winner the candidate that got the most penalized utility points.
        winner = np.argmax(candidate_penalized_utilities, axis=1)

        if isinstance(utilities, torch.Tensor):
            winner = torch.from_numpy(winner).type_as(utilities)

        winner = get_one_hot(winner, n_cands) if one_hot_repr else winner

        return winner, np.ones((len(utilities_np),)).astype(np.bool)
    return egalitarian


@quick_register
def get_kemeny(one_hot=False):
    def kemeny(votes, utilities=None, one_hot_repr=one_hot):
        bs, n_voters, n_cands = votes.shape
        if isinstance(votes, torch.Tensor):
            votes_np = votes.detach().cpu().numpy()
        else:
            votes_np = votes

        cand_position = np.argsort(votes_np, axis=2)

        winners = list()
        for i in range(bs):
            (_, best_rank) = rankaggr_lp(ranks=cand_position[i])
            winner = np.argsort(best_rank)[0]
            winners.append(winner)

        winner = np.array(winners)

        if isinstance(votes, torch.Tensor):
            winner = torch.from_numpy(winner).type_as(votes)
        winner = get_one_hot(winner, n_cands) if one_hot_repr else winner

        unique = np.ones((bs, )).astype(np.bool)

        return winner, unique

    return kemeny


def kendalltau_dist(rank_a, rank_b):
    tau = 0
    n_candidates = len(rank_a)
    for i, j in itertools.combinations(range(n_candidates), 2):
        tau += (np.sign(rank_a[i] - rank_a[j]) ==
                -np.sign(rank_b[i] - rank_b[j]))
    return tau


def rankaggr_brute(ranks):
    min_dist = np.inf
    best_rank = None
    n_voters, n_candidates = ranks.shape
    for candidate_rank in itertools.permutations(range(n_candidates)):
        dist = np.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
        if dist < min_dist:
            min_dist = dist
            best_rank = candidate_rank
    return min_dist, best_rank


def _build_graph(ranks):
    n_voters, n_candidates = ranks.shape
    edge_weights = np.zeros((n_candidates, n_candidates))
    for i, j in itertools.combinations(range(n_candidates), 2):
        preference = ranks[:, i] - ranks[:, j]
        h_ij = np.sum(preference < 0)  # prefers i to j
        h_ji = np.sum(preference > 0)  # prefers j to i
        if h_ij > h_ji:
            edge_weights[i, j] = h_ij - h_ji
        elif h_ij < h_ji:
            edge_weights[j, i] = h_ji - h_ij
    return edge_weights


def rankaggr_lp(ranks):
    """Kemeny-Young optimal rank aggregation"""

    n_voters, n_candidates = ranks.shape

    # maximize c.T * x
    edge_weights = _build_graph(ranks)
    c = -1 * edge_weights.ravel()

    idx = lambda i, j: n_candidates * i + j

    # constraints for every pair
    assert n_candidates % 1 == 0
    pairwise_constraints = np.zeros((int((n_candidates * (n_candidates - 1)) / 2), n_candidates ** 2))
    for row, (i, j) in zip(pairwise_constraints,
                           itertools.combinations(range(n_candidates), 2)):
        row[[idx(i, j), idx(j, i)]] = 1

    # and for every cycle of length 3
    triangle_constraints = np.zeros(((n_candidates * (n_candidates - 1) *
                                      (n_candidates - 2)),
                                     n_candidates ** 2))
    for row, (i, j, k) in zip(triangle_constraints,
                              itertools.permutations(range(n_candidates), 3)):
        row[[idx(i, j), idx(j, k), idx(k, i)]] = 1

    constraints = np.vstack([pairwise_constraints, triangle_constraints])
    constraint_rhs = np.hstack([np.ones(len(pairwise_constraints)),
                                np.ones(len(triangle_constraints))])
    constraint_signs = np.hstack([np.zeros(len(pairwise_constraints)),  # ==
                                  np.ones(len(triangle_constraints))])  # >=

    obj, x, duals = lp_solve(f=c, a=constraints, b=constraint_rhs, e=constraint_signs,
                             xint=range(1, 1 + n_candidates ** 2))

    x = np.array(x).reshape((n_candidates, n_candidates))
    aggr_rank = x.sum(axis=1)

    return obj, aggr_rank


if __name__ == "__main__":
    """
    Run from root. 
    python -m src.voting.voting_rules
    """
    test_num = 0

    if test_num == 0:
        from src.data import get_default_mimicking_loader
        from src.data.datasets.ballot import Ballot
        import time
        import matplotlib.pyplot as plt
        # Test get_kemeny().
        voter_num = 99
        times = list()
        cand_nums = np.arange(5, 29)
        for cand_num in cand_nums:
            blt = Ballot(max_num_voters=voter_num, min_num_voters=voter_num-1, max_num_candidates=cand_num, min_num_candidates=cand_num-1,
                         return_graph=False, remove_ties=False, batch_size=64, epoch_length=256,
                         voting_rule=get_kemeny(), utility_distribution="uniform", one_hot_candidates=True)
            start = time.time()
            rankings, labels, utilities = blt[0]
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            print(f"Cand num: {cand_num}, voter num: {voter_num}, time elapsed: {elapsed:.4f}")

        plt.plot(cand_nums, times, 'o-')
        plt.xlabel("cand num")
        plt.ylabel('time (seconds)')
        plt.title(f"Runtime of Kemeny. num voter: {voter_num}. num cand: varying")
        plt.show()
    if test_num == 1:
        # Test Kemeny results.
        ranks = np.array([[0, 1, 2, 3, 4],
                          [0, 1, 3, 2, 4],
                          [4, 1, 2, 0, 3],
                          [4, 1, 0, 2, 3],
                          [4, 1, 3, 2, 0]])
        votes = np.argsort(ranks, axis=1)[None, ...]
        kemeny = get_kemeny()
        winner, unique = kemeny(votes=votes)
        breakpoint()

