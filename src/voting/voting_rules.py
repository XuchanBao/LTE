from spaghettini import quick_register
import numpy as np
from scipy.stats import mode


@quick_register
def get_plurality():
    def plurality(votes, utilities):
        # Don't use utilities.
        assert utilities is None, "Plurality should not use utilities. "
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
    def borda(votes, utilites):
        raise NotImplementedError

    return borda


@quick_register
def get_oracle():
    def oracle(votes, utilities):
        raise NotImplementedError

    return oracle
