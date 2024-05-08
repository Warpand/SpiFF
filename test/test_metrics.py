from unittest import TestCase, skipUnless

import torch

from spiff.metrics import Histogram


class TestHistogram(TestCase):
    def test_update(self):
        histogram = Histogram(10)
        histogram(torch.Tensor([0.05, 0.95]))
        res = histogram.compute()
        self.assertTrue(torch.all(res == torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])))

    def test_devices(self):
        histogram = Histogram(10).to("cuda")
        histogram(torch.Tensor([0.05, 0.95]).to("cuda"))
        res = histogram.compute().cpu()
        self.assertTrue(torch.all(res == torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])))


    @skipUnless(torch.cuda.is_available(), "Requires cuda.")
    def test_bins(self):
        histogram = Histogram(10).to("cuda")
        self.assertTrue(
            torch.allclose(
                histogram.bins,
                torch.Tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            )
        )
