import os
import dgl
import torch
import numpy as np
import itertools
from torch.utils.data import Dataset
from spaghettini import quick_register


class RealDataFileType:
    STRICT_ORDER_COMPLETE = "soc"
    STRICT_ORDER_INCOMPLETE = "soi"     # not supported yet


@quick_register
class RealDataset(Dataset):
    def __init__(self, path, filename, file_type, voting_rule,
                 one_hot_candidates=False,
                 batch_size=1,
                 epoch_length=None,
                 max_num_voters=None,
                 min_num_voters=None,
                 one_hot_candidate_dim=None,
                 return_graph=False,
                 remove_ties=True
                 ):
        self.path = path
        self.filename = filename
        self.file_type = file_type
        self.voting_rule = voting_rule
        self.one_hot_candidates = one_hot_candidates

        if return_graph:
            assert batch_size == 1
        self.return_graph = return_graph

        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.max_num_voters = max_num_voters
        self.min_num_voters = min_num_voters
        self.one_hot_candidate_dim = one_hot_candidate_dim
        self.remove_ties = remove_ties

        self._read_file()

        self.empty_token = 0
        self.dummy_utility = 2
        self.utility_distribution = "real_data"

    def __len__(self):
        return self.epoch_length

    def _read_file(self):
        if self.file_type == RealDataFileType.STRICT_ORDER_COMPLETE:
            with open(os.path.join(self.path, f"{self.filename}.{self.file_type}"), 'r') as f:
                self.num_candidates = int(f.readline())

                # next num_candidates lines are the candidate names
                self.candidate_names = []
                for cand_i in range(self.num_candidates):
                    self.candidate_names.append(f.readline())

                # (number of voters, _, remaining lines)
                num_voters, _, lines = f.readline().split(',')
                self.num_voters = int(num_voters)
                lines = int(lines)

                # shape = (num_voters, num_candidates)
                self.raw_rankings = np.zeros((self.num_voters, self.num_candidates))
                self.raw_utilities = np.zeros((self.num_voters, self.num_candidates))

                voter_idx = 0
                for line_i in range(lines):
                    line_info = f.readline().split(',')
                    n_voters_line = int(line_info[0])       # num of voters with this preference
                    for cand_i in range(self.num_candidates):
                        # dataset has 1-based indexing, convert to 0-based
                        candidate_id = float(line_info[cand_i + 1]) - 1

                        self.raw_rankings[voter_idx: voter_idx + n_voters_line, cand_i] = candidate_id
                        self.raw_utilities[voter_idx: voter_idx + n_voters_line, int(candidate_id)] = self.num_candidates - cand_i
                    voter_idx += n_voters_line
        else:
            raise NotImplementedError(f"Dataset of filetype {self.file_type} is not supported!")

    def _process_one_hot(self, rankings):
        if self.one_hot_candidates:
            one_hot_rankings = np.array(
                rankings[..., None] == np.arange(self.one_hot_candidate_dim)[None, ...]).astype(np.float)

            rankings_full = self.empty_token * np.ones((rankings.shape[0], rankings.shape[1],
                                                        self.one_hot_candidate_dim, self.one_hot_candidate_dim))
            rankings_full[:, :, :self.num_candidates, :] = one_hot_rankings
        else:
            raise NotImplementedError("Candidates need to be one-hot encoded for now.")

        return rankings_full

    def subset_raw_rankings_and_utilities(self, num_voters):
        voter_ids = np.random.choice(self.num_voters, size=num_voters)
        raw_ranking_subset = self.raw_rankings[voter_ids, :]
        raw_utility_subset = self.raw_utilities[voter_ids, :]
        return raw_ranking_subset, raw_utility_subset

    def __getitem__(self, idx):
        num_voters = np.random.choice(np.arange(self.min_num_voters, self.max_num_voters + 1))   # [min, max + 1)

        success = False
        while not success:
            rankings = np.zeros((self.batch_size, num_voters, self.num_candidates))
            utilities = np.zeros((self.batch_size, num_voters, self.num_candidates))

            for batch_i in range(self.batch_size):
                rankings[batch_i], utilities[batch_i] = self.subset_raw_rankings_and_utilities(num_voters)

            winner, unique = self.voting_rule(rankings, utilities=None)

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

        rankings_full = self._process_one_hot(rankings)

        # Move to torch tensors.
        # flatten the last 2 dims
        xs_torch = torch.tensor(rankings_full).float().view(rankings.shape[0], rankings.shape[1], -1)
        ys_torch = torch.tensor(winner).long()

        utilities_full = self.empty_token * torch.ones(
            (rankings.shape[0], rankings.shape[1], self.one_hot_candidate_dim))
        utilities_full[:, :, :self.num_candidates] = torch.tensor(utilities)

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
            utilities_fuller = np.zeros((utilities_full.shape[0], self.max_num_voters, self.one_hot_candidate_dim))
            utilities_fuller[:, :num_voters, :] = utilities_full
            utilities_full = utilities_fuller.squeeze(0)  # get rid of the first dummy batch size dimension

        # Return the rankings and the winners.
        return xs_torch, ys_torch, utilities_full
