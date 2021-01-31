from spaghettini import quick_register

from src.data.datasets.ballot import Ballot


@quick_register
def get_default_mimicking_loader(distribution, voting_rule, batch_size, return_graph):
    return Ballot(max_num_voters=99, min_num_voters=2, max_num_candidates=29, min_num_candidates=2,
                  batch_size=batch_size, epoch_length=256, voting_rule=voting_rule, utility_distribution=distribution,
                  one_hot_candidates=True, one_hot_candidate_dim=None, return_graph=return_graph, remove_ties=True)


