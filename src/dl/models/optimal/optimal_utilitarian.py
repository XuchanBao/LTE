from spaghettini import quick_register
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init

from src.utils.voting_utils import get_one_hot


@quick_register
class OptimalUtilitarian(nn.Module):
    def __init__(self, num_candidates):
        super().__init__()
        self.num_candidates = num_candidates

        # Initialize the scoring function.
        self.scores = torch.Tensor(num_candidates).float()
        self.n = 0
        self._init_scores()

        # Initialize dummy parameter so that the optimizer doesn't complain.
        self.dummy_param = Parameter(torch.Tensor(1))
        init.zeros_(self.dummy_param)

    def _init_scores(self):
        init.zeros_(self.scores)

    def forward(self, xs, utilities, **kwargs):
        # ____ Pick the winner. ____
        winner = pick_score_based_winner(rankings=xs, scores=self.scores, return_one_hot=True)

        # ___ Update scores. ____
        # Sort the utilities.
        sorted_utils = torch.sort(utilities, descending=True)[0]

        # Update the scores.
        dn = sorted_utils.shape[0] * sorted_utils.shape[1]  # Number of samples used for updating expectation.
        ds = sorted_utils.sum(dim=[0, 1])
        self.scores = ((self.n / (self.n + dn)) * self.scores + (1. / (self.n + dn)) * ds).float()
        self.n = self.n + dn
        print(f"\n n: {self.n}, mean: {self.scores.mean()}, std: {self.scores.std()}, median: {self.scores[10]}")

        # Trick pytorch autograd into backpropping through this dummy computational graph.
        winner = winner.detach().float() + torch.zeros_like(winner) * self.dummy_param

        return winner


def pick_score_based_winner(rankings, scores, return_one_hot=True):
    bs, num_voters, num_candidates = rankings.shape
    all_cand_scores = torch.zeros((bs, num_candidates))
    for c_id in range(num_candidates):
        pos = (rankings == c_id)
        curr_cand_score = (pos.float() @ scores[:, None]).sum(dim=1).squeeze()
        all_cand_scores[:, c_id] = curr_cand_score
    winner = torch.argmax(all_cand_scores, dim=1)
    if return_one_hot:
        winner = get_one_hot(winner, num_candidates=num_candidates)
    return winner


def pick_score_based_winner_alternative(rankings, scores, return_one_hot=True):
    num_candidates = rankings.shape[-1]
    candidate_ids = torch.arange(num_candidates)
    counts = (rankings[..., None] == candidate_ids[None, None, None, :]).sum(dim=1).float()
    candidate_scores = (counts.transpose(1, 2) @ scores[None, :, None]).squeeze()
    winner = torch.argmax(candidate_scores, dim=1)
    if return_one_hot:
        winner = get_one_hot(winner, num_candidates=num_candidates)
    return winner


if __name__ == "__main__":
    """
    Run from root. 
    python -m src.dl.models.optimal.optimal_utilitarian
    """
    test_num = 1

    if test_num == 0:
        # Check if it is possible to store inputs as attributes of pytorch tensors.
        sf = OptimalUtilitarian(num_candidates=5)

    if test_num == 1:
        # Check if score based winner implementation is correct.
        from src.data.datasets.ballot import Ballot
        from src.voting.voting_rules import get_plurality

        blt = Ballot(max_num_voters=99, min_num_voters=50, max_num_candidates=20, min_num_candidates=20,
                     return_graph=False, remove_ties=False, batch_size=2048, epoch_length=256,
                     voting_rule=get_plurality(), utility_distribution="uniform", one_hot_candidates=False)
        rankings_, labels, utilities = blt[0]

        # Initialize scores.
        scores_ = torch.rand(size=(20,))

        # Pick winner the fast way.
        winner_fast = pick_score_based_winner(rankings=rankings_, scores=scores_)

        # Pick winner the slow way.
        winner_slow = pick_score_based_winner_alternative(rankings=rankings_, scores=scores_)

        # Check if the answers match.
        print((winner_fast == winner_slow).all())
