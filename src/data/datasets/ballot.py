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
            dirichlet_alphas = np.random.choice([2., 3.], size=(num_candidates,))
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
            ranking_onehot = np.array(rankings[..., None] == np.arange(self.one_hot_candidate_dim)[None, ...]).astype(
                np.float)
            rankings_full = self.empty_token * np.ones(
                (rankings.shape[0], rankings.shape[1], self.max_num_candidates, self.one_hot_candidate_dim))
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


def _check_strong_pairwise_transitivity_in_election(votes):
    assert votes.shape[0] == 1
    # ____ Get pairwise comparison matrix. ____
    if isinstance(votes, torch.Tensor):
        votes_np = votes.detach().cpu().numpy()
    else:
        votes_np = votes

    bs, n_voters, n_cands = votes_np.shape

    cand_position = np.argsort(votes_np, axis=2)

    pairwise_win_probs = np.zeros((bs, n_cands, n_cands))
    for cand_a in range(n_cands):
        for cand_b in range(cand_a + 1, n_cands):
            num_a_win_b = np.sum(cand_position[:, :, cand_a] < cand_position[:, :, cand_b], axis=1)
            pairwise_win_probs[:, cand_a, cand_b] = float(num_a_win_b) / n_voters
            pairwise_win_probs[:, cand_b, cand_a] = float((n_voters - num_a_win_b)) / n_voters

    # ____ Check for strong pairwise transitivity. ____
    cand_ids = np.arange(0, n_cands)
    satisfies_spt_count, satisfies_strict_spt_count, satisfies_premise_count, total_count = 0, 0, 0, 0
    for i, j, k in itertools.product(cand_ids, cand_ids, cand_ids):
        if i == j or j == k or i == k:
            continue
        pij = pairwise_win_probs[0, i, j]  # Probability i beats j.
        pjk = pairwise_win_probs[0, j, k]
        pik = pairwise_win_probs[0, i, k]
        premise_satisfied = pij >= 0.5 and pjk >= 0.5
        spt_satisfied = not premise_satisfied or pik >= max(pij, pjk)
        strict_spt_satisfied = premise_satisfied and pik >= max(pij, pjk)

        # Record.
        if premise_satisfied:
            satisfies_premise_count += 1
        if spt_satisfied:
            satisfies_spt_count += 1
        if strict_spt_satisfied:
            satisfies_strict_spt_count += 1
        total_count += 1

    # Extract final statistics.
    ratio_of_spt_triplets_to_all_triplets = satisfies_spt_count / total_count
    ratio_of_strict_spt_triplets_to_premise_triplets = satisfies_strict_spt_count / satisfies_premise_count

    return ratio_of_spt_triplets_to_all_triplets, ratio_of_strict_spt_triplets_to_premise_triplets


def _check_strong_pairwise_transitivity_in_ballot(ballot, num_elections=100):
    assert ballot.batch_size == 1
    ratio_of_spt_to_all_list, ratio_of_strict_spt_to_premise_list = list(), list()
    for i in range(num_elections):
        rankings, _, _ = blt[0]
        ratio_of_spt_to_all, ratio_of_strict_spt_to_premise = _check_strong_pairwise_transitivity_in_election(votes=rankings)
        ratio_of_spt_to_all_list.append(ratio_of_spt_to_all)
        ratio_of_strict_spt_to_premise_list.append(ratio_of_strict_spt_to_premise)

    ratio_of_spt_to_all_np = np.array(ratio_of_spt_to_all_list)
    ratio_of_strict_spt_to_premise_np = np.array(ratio_of_strict_spt_to_premise_list)
    print(f"==== Analyzing for {ballot.utility_distribution}. ====")
    print(f"Stats of spt to all triplets: "
          f"mean of {ratio_of_spt_to_all_np.mean():.3f}, "
          f"std of {ratio_of_spt_to_all_np.std():.3f}, "
          f"max of {ratio_of_spt_to_all_np.max():.3f}")
    print(f"Stats of strict spt to premise triplets: "
          f"mean of {ratio_of_strict_spt_to_premise_np.mean():.3f}, "
          f"std of {ratio_of_strict_spt_to_premise_np.std():.3f}"
          f"max of {ratio_of_strict_spt_to_premise_np.max():.3f}")


if __name__ == "__main__":
    """
    Run from root. 
    python -m src.data.datasets.ballot
    """
    test_num = 3

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
                     voting_rule=get_plurality(), utility_distribution="uniform", one_hot_candidates=True)
        rankings, labels, utilities = blt[0]
        utils_flt = utilities.reshape(-1)
        plt.hist(utils_flt, bins=50)
        plt.show()
        breakpoint()

    if test_num == 3:
        # Check for strong pairwise transitivity for various Dirichlet parameters.
        from src.voting.voting_rules import get_plurality
        import matplotlib.pyplot as plt
        from src.data import get_default_mimicking_loader

        distributions = ["uniform", "indecisive", "landslide", "polarized", "skewed"]
        for dist in distributions:
            loader = get_default_mimicking_loader(distribution=dist, voting_rule=get_plurality(), return_graph=False)
            blt = loader.dataset
            blt.one_hot_candidates = False
            blt.batch_size = 1
            _check_strong_pairwise_transitivity_in_ballot(ballot=blt, num_elections=1000)


