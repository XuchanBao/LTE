from spaghettini import quick_register

from torch.utils.data import DataLoader

import src.data.data_loading
import src.data.datasets

from src.data.datasets.ballot import Ballot
from src.data.data_loading.collates import dgl_ballot_collate, batched_ballot_collate
from src.utils.misc import seed_workers


@quick_register
def get_default_mimicking_loader(distribution, voting_rule, return_graph):
    max_num_voters = 99
    min_num_voters = 2
    max_num_candidates = 29
    min_num_candidates = 2
    batch_size = 64
    epoch_len = 8
    one_hot_candidates = True
    one_hot_candidate_dim = None
    remove_ties = True
    num_workers = 10
    if return_graph is True:
        dataset = Ballot(max_num_voters=max_num_voters, min_num_voters=min_num_voters,
                         max_num_candidates=max_num_candidates, min_num_candidates=min_num_candidates,
                         batch_size=1, epoch_length=epoch_len, voting_rule=voting_rule,
                         utility_distribution=distribution, one_hot_candidates=one_hot_candidates,
                         one_hot_candidate_dim=one_hot_candidate_dim, return_graph=True,
                         remove_ties=remove_ties)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          collate_fn=dgl_ballot_collate, worker_init_fn=seed_workers)
    else:
        dataset = Ballot(max_num_voters=max_num_voters, min_num_voters=min_num_voters,
                         max_num_candidates=max_num_candidates, min_num_candidates=min_num_candidates,
                         batch_size=batch_size, epoch_length=epoch_len, voting_rule=voting_rule,
                         utility_distribution=distribution, one_hot_candidates=one_hot_candidates,
                         one_hot_candidate_dim=one_hot_candidate_dim, return_graph=False,
                         remove_ties=remove_ties)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                          collate_fn=batched_ballot_collate, worker_init_fn=seed_workers)


@quick_register
def get_both_default_mimicking_loaders(distribution, voting_rule, return_graph):
    max_num_voters = 99
    min_num_voters = 2
    max_num_candidates = 29
    min_num_candidates = 2
    batch_size = 64
    epoch_len = 8
    one_hot_candidates = True
    one_hot_candidate_dim = None
    remove_ties = True
    num_workers = 10
    if return_graph is True:
        dataset = Ballot(max_num_voters=max_num_voters, min_num_voters=min_num_voters,
                         max_num_candidates=max_num_candidates, min_num_candidates=min_num_candidates,
                         batch_size=1, epoch_length=batch_size*epoch_len, voting_rule=voting_rule,
                         utility_distribution=distribution, one_hot_candidates=one_hot_candidates,
                         one_hot_candidate_dim=one_hot_candidate_dim, return_graph=True,
                         remove_ties=remove_ties)
        dl1 = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         collate_fn=dgl_ballot_collate, worker_init_fn=seed_workers)
        dl2 = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         collate_fn=dgl_ballot_collate, worker_init_fn=seed_workers)
        return dl1, dl2
    else:
        dataset = Ballot(max_num_voters=max_num_voters, min_num_voters=min_num_voters,
                         max_num_candidates=max_num_candidates, min_num_candidates=min_num_candidates,
                         batch_size=batch_size, epoch_length=epoch_len, voting_rule=voting_rule,
                         utility_distribution=distribution, one_hot_candidates=one_hot_candidates,
                         one_hot_candidate_dim=one_hot_candidate_dim, return_graph=False,
                         remove_ties=remove_ties)
        dl1 = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                         collate_fn=batched_ballot_collate, worker_init_fn=seed_workers)
        dl2 = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                         collate_fn=batched_ballot_collate, worker_init_fn=seed_workers)
        return dl1, dl2
