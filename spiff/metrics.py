import torch
import torchmetrics


class Histogram(torchmetrics.Metric):
    """Metric computing histograms."""

    def __init__(self, bins: torch.Tensor) -> None:
        """
        Construct the metric.

        :param bins: bins for calculating histogram.
        """
        super().__init__()

        self.register_buffer("_bins", bins)
        self.add_state("hist", default=torch.zeros(len(bins)), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor) -> None:
        """
        Update the histogram state with new values.

        :param values: tensor of new values.
        """

        histogram = torch.histogram(values, bins=self._bins)
        self.hist += histogram.hist

    def compute(self) -> torch.Tensor:
        """
        Return the current histogram values.

        :return: histogram values.
        """

        return self.hist

    @property
    def bins(self):
        return self._bins
