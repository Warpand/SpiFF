import torch
import torchmetrics


class Histogram(torchmetrics.Metric):
    """Metric computing histograms."""

    def __init__(self, num_bins: int, left: float = 0.0, right: float = 1.0) -> None:
        """
        Construct the metric.

        :param num_bins: number of bins for the histogram calculation.
        :param left: lower bound of the lowest values bin.
        :param right: upped bound of the greatest values bin.
        """

        super().__init__()

        self.num_bins = num_bins
        self.left = left
        self.right = right
        self.add_state("hist", default=torch.zeros(num_bins), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor) -> None:
        """
        Update the histogram state with new values.

        :param values: tensor of new values.
        """

        histogram = torch.histc(values, self.num_bins, self.left, self.right)
        self.hist += histogram

    def compute(self) -> torch.Tensor:
        """
        Return the current histogram values.

        :return: histogram values.
        """

        return self.hist

    @property
    def bins(self):
        return torch.linspace(self.left, self.right, self.num_bins + 1)
