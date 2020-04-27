from spaghettini import quick_register
import numpy as np
from scipy.stats import mode
import torch

from src.utils.voting_utils import get_one_hot


@quick_register
def get_plurality(one_hot=False):
    def plurality(votes, utilities=None, one_hot_repr=one_hot):
        # Don't use utilities.
        utilities = None

        # votes: (batch_size, # of voters, # of candidates)
        # TODO: Decide how to handle ties.
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
    def borda(votes, utilites=None, one_hot_repr=one_hot):
        breakpoint()

        # Don't use utilities.
        utilities = None

        # ____ Compute borda scores for each candidate. ____
        winner = None
        unique = None
        pass

        return winner, unique

    return borda


@quick_register
def get_oracle(one_hot=False):
    def oracle(votes, utilities, one_hot_repr=one_hot):
        # Don't use votes.
        votes = None

        # Get the total amount of utility assigned to each candidate.
        candidate_utilities = utilities.sum(axis=1)

        # Declare as winner the candidate that got the most utility points.
        winner = np.argmax(candidate_utilities, axis=1)

        # Get one hot representation is asked.
        winner = get_one_hot(winner) if one_hot_repr else winner

        return winner, np.ones((len(utilities), )).astype(np.bool)

    return oracle
