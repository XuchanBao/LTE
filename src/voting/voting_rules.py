from spaghettini import quick_register
import numpy as np
from scipy.stats import mode


@quick_register
def get_plurality():
    def plurality(votes, utilities=None):
        # Don't use utilities.
        utilities = None

        # votes: (batch_size, # of voters, # of candidates)
        # TODO: Decide how to handle ties.
        # Select the top votes of the voters.
        top_votes = votes[:, :, 0]

        # Return the most popular candidate.
        winner = mode(top_votes, axis=1).mode

        return winner.squeeze()
    return plurality


@quick_register
def get_borda():
    def borda(votes, utilites=None):
        # Don't use utilities.
        utilities = None

        raise NotImplementedError

    return borda


@quick_register
def get_oracle():
    def oracle(votes, utilities):
        # Don't use votes.
        votes = None

        # Get the total amount of utility assigned to each candidate.
        candidate_utilities = utilities.sum(axis=1)

        # Declare as winner the candidate that got the most utility points.
        winner = np.argmax(candidate_utilities, axis=1)

        return winner

    return oracle
