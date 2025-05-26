"""Classes for specifying probe pytorch modules."""

import torch.nn as nn
import torch
import torch.nn.functional as F


class Probe(nn.Module):
    pass


class OneWordPSDProbe(Probe):
    """Computes squared L2 norm of words after projection by a matrix."""

    def __init__(self, args):
        print("Constructing OneWordPSDProbe")
        super(OneWordPSDProbe, self).__init__()
        self.args = args
        self.probe_rank = args["probe"]["maximum_rank"]
        self.model_dim = args["model"]["hidden_dim"]
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n depths after projection
        for each sentence in a batch.

        Computes (Bh_i)^T(Bh_i) for all i

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of depths of shape (batch_size, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        norms = torch.bmm(
            transformed.view(batchlen * seqlen, 1, rank),
            transformed.view(batchlen * seqlen, rank, 1),
        )
        norms = norms.view(batchlen, seqlen)
        return norms


class OneWordNonPSDProbe(Probe):
    """Computes a bilinear affinity between each word representation and itself.

    This is different from the probes in A Structural Probe... as the
    matrix in the quadratic form is not guaranteed positive semi-definite

    """

    def __init__(self, args):
        print("Constructing OneWordNonPSDProbe")
        super(OneWordNonPSDProbe, self).__init__()
        self.args = args
        self.model_dim = args["model"]["hidden_dim"]
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.model_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n depths after projection
        for each sentence in a batch.

        Computes (h_i^T)A(h_i) for all i

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of depths of shape (batch_size, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = batch.size()
        norms = torch.bmm(
            transformed.view(batchlen * seqlen, 1, rank),
            batch.view(batchlen * seqlen, rank, 1),
        )
        norms = norms.view(batchlen, seqlen)
        return norms


class TwoWordPSDProbe(Probe):
    """Computes squared L2 distance after projection by a matrix.

    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, args):
        print("Constructing TwoWordPSDProbe")
        super(TwoWordPSDProbe, self).__init__()
        self.args = args
        self.probe_rank = args["probe"]["maximum_rank"]
        self.model_dim = args["model"]["hidden_dim"]
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n^2 pairs of distances after projection
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


class TwoWordNonPSDProbe(Probe):
    """Computes a bilinear function of difference vectors.

    For a batch of sentences, computes all n^2 pairs of scores
    for each sentence in the batch.
    """

    def __init__(self, args):
        print("TwoWordNonPSDProbe")
        super(TwoWordNonPSDProbe, self).__init__()
        self.args = args
        self.probe_rank = args["probe"]["maximum_rank"]
        self.model_dim = args["model"]["hidden_dim"]
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.model_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n^2 pairs of difference scores
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.
        Computes (h_i-h_j)^TA(h_i-h_j) for all i,j

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
        """
        batchlen, seqlen, rank = batch.size()
        batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
        diffs = (batch_square - batch_square.transpose(1, 2)).view(
            batchlen * seqlen * seqlen, rank
        )
        psd_transformed = torch.matmul(diffs, self.proj).view(
            batchlen * seqlen * seqlen, 1, rank
        )
        dists = torch.bmm(
            psd_transformed, diffs.view(batchlen * seqlen * seqlen, rank, 1)
        )
        dists = dists.view(batchlen, seqlen, seqlen)
        return dists


class TwoWordPSDCumulativeProbe(Probe):
    """Computes squared L2 distance after projection by a matrix.

    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, args):
        print("Constructing TwoWordPSDCumulativeProbe")
        super().__init__()
        self.args = args
        self.probe_rank = args["probe"]["maximum_rank"]
        self.model_dim = args["model"]["hidden_dim"]
        self.num_layers = args["model"]["model_layer"] + 1
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
        # prepare scaling parameter(scalar), and learnable weights for each layer
        self.weights = nn.Parameter(data=torch.zeros(self.num_layers))
        self.scaling = nn.Parameter(data=torch.zeros(1))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        nn.init.uniform_(self.weights, -0.05, 0.05)
        nn.init.uniform_(self.scaling, -0.05, 0.05)
        self.to(args["device"])

    def center_of_gravity(self):
        softmax_weights = F.softmax(self.weights, dim=0)
        return torch.sum(
            torch.arange(self.num_layers).to(self.args["device"]) * softmax_weights
        ).item()

    def forward(self, batch):
        """Computes all n^2 pairs of distances after projection
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j

        Args:
          batch: a batch of word representations of the shape
            (batch_size, num_layers, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        batchlen, num_layers, seqlen, rank = batch.size()
        batch_weighted = (
            torch.einsum("blsr,l->bsr", batch, F.softmax(self.weights, dim=0))
            * self.scaling
        )
        assert batch_weighted.size() == (batchlen, seqlen, rank)
        transformed = torch.matmul(batch_weighted, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)  # (b, s, 1, r)
        transformed = transformed.expand(-1, -1, seqlen, -1)  # (b, s, s, r)
        transposed = transformed.transpose(1, 2)  # (b, s, s, r)
        diffs = transformed - transposed  # (b, s, s, r)
        squared_diffs = diffs.pow(2)  # (b, s, s, r)
        squared_distances = torch.sum(squared_diffs, -1)  # (b, s, s)
        return squared_distances
