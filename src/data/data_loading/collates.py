import dgl
import torch
from spaghettini import quick_register


@quick_register
def dgl_ballot_collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label, utilities).
    graphs, labels, utilities = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), torch.tensor(utilities)


# TODO: test the following
@quick_register
def batched_ballot_collate(samples):
    assert len(samples) == 1, "Samples should be list of length 1!"
    rankings, labels, utilities = samples[0]
    return rankings, labels, torch.tensor(utilities)
