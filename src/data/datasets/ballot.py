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
                 utility_distribution="uniform"):
        self.max_num_voters = max_num_voters
        self.max_num_candidates = max_num_candidates
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.utility_distribution = utility_distribution
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
            print("Please specify a valid distribution over utilies. Currentl only 'uniform' is supported.")
            exit(-1)

        # Get the rankings of the voters (descending order).
        rankings = np.argsort(utilities, axis=2)[:, :, ::-1]

        # Pick the winner.
        winner = self.voting_rule(rankings, utilities=None)

        # Add "dummy" rankings to make sure all rankings have the same dimensionality (i.e. max_num_candidates).
        rankings_full = np.zeros((rankings.shape[0], rankings.shape[1], self.max_num_candidates))
        rankings_full[:, :, :num_candidates] = rankings
        rankings_full[:, :, num_candidates:] = self.empty_token

        # Move to torch tensors.
        xs_torch = torch.tensor(rankings_full).float()
        ys_torch = torch.tensor(winner).long()

        # Return the rankings and the winners.
        return xs_torch, ys_torch


if __name__ == "__main__":
    def test_dirichlet():
        import matplotlib.pyplot as plt
        samples = np.random.dirichlet([1,1], size=(100,))
        plt.scatter(samples[:, 0], samples[:, 1])
        plt.show()

    # test_dirichlet()

    def test_plurality():
        plurality_dataset = Ballot()
        plurality_dataset.__getitem__(0)

    test_plurality()
