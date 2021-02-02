from spaghettini import quick_register
import numpy as np
from scipy.stats import mode
import torch

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
