import numpy as np
import torch
from torch.utils.data import Dataset
from spaghettini import quick_register

from src.voting.voting_rules import get_plurality


@quick_register
class Ballot(Dataset):
    def __init__(self,
                 max_num_voters=20,
                 max_num_candidates=10,
                 batch_size=32,
                 epoch_length=256,
                 voting_rule=get_plurality(),
                 utility_distribution="uniform",
                 one_hot_candidates=False):
        self.max_num_voters = max_num_voters
        self.max_num_candidates = max_num_candidates
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.utility_distribution = utility_distribution
        self.one_hot_candidates = one_hot_candidates
        self.voting_rule = voting_rule

        self.empty_token = -1

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        # Sample number of voters and candidates.
        num_voters = np.random.randint(1, self.max_num_voters + 1)
        num_candidates = np.random.randint(1, self.max_num_candidates + 1)

        # Sample utility profiles.
        if self.utility_distribution == "uniform":
            dirichlet_alphas = np.ones(num_candidates)
            utilities = np.random.dirichlet(alpha=dirichlet_alphas, size=(self.batch_size, num_voters))
        else:
            raise ValueError("Please specify a valid distribution over utilities. "
                             "Currently only 'uniform' is supported.")

        # Get the rankings of the voters (descending order).
        rankings = np.argsort(utilities, axis=2)[:, :, ::-1]

        # Pick the winner.
        winner, unique = self.voting_rule(rankings, utilities=utilities)

        # remove rows with ties
        tied_rows = np.argwhere(unique == False).squeeze()
        winner = np.delete(winner, tied_rows, 0)
        rankings = np.delete(rankings, tied_rows, 0)
        utilities = np.delete(utilities, tied_rows, 0)

        if not self.one_hot_candidates:
            # Add "dummy" rankings to make sure all rankings have the same dimensionality (i.e. max_num_candidates).
            rankings_full = np.zeros((rankings.shape[0], rankings.shape[1], self.max_num_candidates))
            rankings_full[:, :, :num_candidates] = rankings
            rankings_full[:, :, num_candidates:] = self.empty_token
        else:
            # ranking_onehot = np.zeros(rankings.shape + (num_candidates, ))
            ranking_onehot = np.array(rankings[..., None] == np.arange(num_candidates)[None, ...]).astype(np.float)
            rankings_full = self.empty_token * np.ones((rankings.shape[0], rankings.shape[1],
                                                       self.max_num_candidates, self.max_num_candidates))
            rankings_full[:, :, :num_candidates, :num_candidates] = ranking_onehot

        # Add "dummy" utilities to make sure all utilities have the same dimensionality.
        utilities_full = np.zeros((utilities.shape[0], utilities.shape[1], self.max_num_candidates))
        utilities_full[:, :, :num_candidates] = utilities
        utilities_full[:, :, num_candidates:] = 0.0

        # Move to torch tensors.
        xs_torch = torch.tensor(rankings_full).float().view(rankings.shape[0], rankings.shape[1], -1)
        ys_torch = torch.tensor(winner).long()

        # Return the rankings and the winners.
        return xs_torch, ys_torch, utilities_full


if __name__ == "__main__":
    def test_dirichlet():
        import matplotlib.pyplot as plt
        samples = np.random.dirichlet([1, 1], size=(100,))
        plt.scatter(samples[:, 0], samples[:, 1])
        plt.show()


    # test_dirichlet()

    def test_plurality():
        plurality_dataset = Ballot()
        plurality_dataset.__getitem__(0)


    test_plurality()
