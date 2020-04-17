import torch


def compute_accuracy(logits, scalar_targets):
    return (logits.argmax(axis=1) == scalar_targets).float().mean()


def compute_distortion_ratios(logits, utilities):
    bs = logits.shape[0]  # Batch size.

    # ____ Get the best possible total utility values. ____
    candidate_utilities = utilities.sum(axis=1)
    best_utilities, _ = torch.max(candidate_utilities, axis=1)

    # ____ Get the total utility obtained by the voting rule. ____
    elected_candidates = logits.argmax(axis=1)
    obtained_utilities = candidate_utilities[torch.arange(bs), elected_candidates]

    # ____ Compute the inverse distortion ratios. ____
    inv_distortion_ratios = (obtained_utilities / best_utilities).cpu().numpy()

    return inv_distortion_ratios
