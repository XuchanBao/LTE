import numpy as np
import torch
import dgl
import itertools
from torch.utils.data import Dataset
from spaghettini import quick_register

from src.voting.voting_rules import get_plurality


@quick_register
class Ballot(Dataset):
    def __init__(self,
                 max_num_voters=20,
                 max_num_candidates=20,
                 batch_size=1,
                 epoch_length=256,
                 voting_rule=get_plurality(),
                 utility_distribution="uniform",
                 one_hot_candidates=False,
                 one_hot_candidate_dim=None,
                 min_num_voters=10,
                 min_num_candidates=10,
                 return_graph=False,
                 remove_ties=True):
        if return_graph:
            assert batch_size == 1
        self.min_num_voters = min_num_voters
        self.max_num_voters = max_num_voters

        if isinstance(self.min_num_voters, list):
            assert isinstance(self.max_num_voters, list)
            assert len(self.min_num_voters) == len(self.max_num_voters), \
                "Length for min_num_voters must match max_num_voters"
            self.voter_num_list = []
            for interval_i in range(len(self.min_num_voters)):
                self.voter_num_list.extend(list(range(self.min_num_voters[interval_i],
                                                      self.max_num_voters[interval_i] + 1)))
        else:
            self.voter_num_list = list(range(self.min_num_voters, self.max_num_voters))
        self.min_num_candidates = min_num_candidates
        self.max_num_candidates = max_num_candidates
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.utility_distribution = utility_distribution
        self.one_hot_candidates = one_hot_candidates
        self.voting_rule = voting_rule
        self.return_graph = return_graph
        self.remove_ties = remove_ties
        self.one_hot_candidate_dim = self.max_num_candidates if one_hot_candidate_dim is None else one_hot_candidate_dim
        assert self.one_hot_candidate_dim >= self.max_num_candidates

        self.empty_token = 0

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        # Sample number of voters and candidates.
        # num_voters = np.random.randint(self.min_num_voters, self.max_num_voters + 1)
        num_voters = np.random.choice(self.voter_num_list)
        num_candidates = np.random.randint(self.min_num_candidates, self.max_num_candidates + 1)

        # Sample utility profiles.
        # uniform, indecisive, landslide, polarized, skewed
        if self.utility_distribution == "uniform":
            dirichlet_alphas = np.ones(num_candidates)
        elif self.utility_distribution == "indecisive":
            dirichlet_alphas = 2.0 * np.ones(num_candidates)
        elif self.utility_distribution == "landslide":
            dirichlet_alphas = np.ones(num_candidates)
            preferred_cand = np.random.choice(num_candidates)
            dirichlet_alphas[preferred_cand] = 3.0
        elif self.utility_distribution == "polarized":
            dirichlet_alphas = 0.5 * np.ones(num_candidates)
        elif self.utility_distribution == "skewed":
            dirichlet_alphas = np.random.choice([2., 3.], size=(num_candidates, ))
        else:
            raise ValueError("Please specify a valid distribution over utilities. "
                             "Must be one of ('uniform', 'indecisive', 'landslide', 'polarized', 'skewed')")

        success = False
        while not success:
            utilities = np.random.dirichlet(alpha=dirichlet_alphas, size=(self.batch_size, num_voters))

            # Get the rankings of the voters (descending order).
            rankings = np.argsort(utilities, axis=2)[:, :, ::-1]

            # Pick the winner.
            winner, unique = self.voting_rule(rankings, utilities=utilities)

            # Remove rows with ties.
            if self.remove_ties:
                tied_rows = np.argwhere(unique == False).squeeze()
                winner = np.delete(winner, tied_rows, 0)
                rankings = np.delete(rankings, tied_rows, 0)
                utilities = np.delete(utilities, tied_rows, 0)

            # Declare success when not returning graph, or when the data point doesn't contain ties.
            if self.return_graph:
                if unique.all() or not self.remove_ties:
                    success = True
            else:
                if unique.any() or not self.remove_ties:
                    success = True

        if not self.one_hot_candidates:
            # Add "dummy" rankings to make sure all rankings have the same dimensionality (i.e. max_num_candidates).
            rankings_full = np.zeros((rankings.shape[0], rankings.shape[1], self.max_num_candidates))
            rankings_full[:, :, :num_candidates] = rankings
            rankings_full[:, :, num_candidates:] = self.empty_token
        else:
            # ranking_onehot = np.zeros(rankings.shape + (num_candidates, ))
            ranking_onehot = np.array(rankings[..., None] == np.arange(self.one_hot_candidate_dim)[None, ...]).astype(np.float)
            rankings_full = self.empty_token * np.ones((rankings.shape[0], rankings.shape[1], self.max_num_candidates, self.one_hot_candidate_dim))
            rankings_full[:, :, :num_candidates, :self.one_hot_candidate_dim] = ranking_onehot

        # Add "dummy" utilities to make sure all utilities have the same dimensionality.
        utilities_full = np.zeros((utilities.shape[0], utilities.shape[1], self.max_num_candidates))
        utilities_full[:, :, :num_candidates] = utilities

        # Move to torch tensors.
        xs_torch = torch.tensor(rankings_full).float().view(rankings.shape[0], rankings.shape[1], -1)
        ys_torch = torch.tensor(winner).long()

        # Build a graph.
        if self.return_graph:
            # Build edges (2, n_voters * n_voters).
            edges = np.array(list(itertools.product(np.arange(num_voters), np.arange(num_voters)))).T

            # Build graph
            graph = dgl.graph((edges[0], edges[1]))

            # Add node features (n_voters, n_cand).
            graph.ndata['feat'] = xs_torch.squeeze(0)

            # Should remove self edges.
            dgl.to_simple(graph, copy_ndata=True)

            xs_torch = graph

            # Pad the utilities_full (num_voters, max_num_cand) --> (max_num_voters, max_num_cand)
            utilities_fuller = np.zeros((utilities_full.shape[0], self.max_num_voters, self.max_num_candidates))
            utilities_fuller[:, :num_voters, :] = utilities_full
            utilities_full = utilities_fuller.squeeze(0)  # get rid of the first dummy batch size dimension

        # Return the rankings and the winners.
        return xs_torch, ys_torch, utilities_full


if __name__ == "__main__":
    """
    Run from root. 
    python -m src.data.datasets.ballot
    """
    test_num = 2

    if test_num == 0:
        def test_dirichlet():
            import matplotlib.pyplot as plt
            samples = np.random.dirichlet([1, 1], size=(100,))
            plt.scatter(samples[:, 0], samples[:, 1])
            plt.show()


        test_dirichlet()

    if test_num == 1:
        def test_plurality():
            plurality_dataset = Ballot()
            plurality_dataset.__getitem__(0)


        test_plurality()

    if test_num == 2:
        from src.voting.voting_rules import get_plurality
        import matplotlib.pyplot as plt

        blt = Ballot(max_num_voters=99, min_num_voters=50, max_num_candidates=20, min_num_candidates=20,
                     return_graph=False, remove_ties=False, batch_size=2048, epoch_length=256,
                     voting_rule=get_plurality(), utility_distribution="uniform_on_interval", one_hot_candidates=True)
        rankings, labels, utilities = blt[0]
        utils_flt = utilities.reshape(-1)
        plt.hist(utils_flt, bins=50)
        plt.show()
        breakpoint()
