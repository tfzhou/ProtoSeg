import torch
import torch.nn.functional as F


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    Q = torch.exp(out / epsilon).t() # K x B
    B = Q.shape[1]
    K = Q.shape[0]

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    Q = Q.t()

    indexs = torch.argmax(Q, dim=1)
    # Q = torch.nn.functional.one_hot(indexs, num_classes=Q.shape[1]).float()
    Q = F.gumbel_softmax(Q, tau=0.5, hard=True)

    return Q, indexs