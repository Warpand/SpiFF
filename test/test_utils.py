from unittest import TestCase

import spiff.utils as utils


class TestUtils(TestCase):
    def test_triple_wise(self):
        it = [1, 2, 3, 4, 5, 6, 7]
        triples = iter(utils.triple_wise(it))
        self.assertTupleEqual((1, 2, 3), next(triples))
        self.assertTupleEqual((4, 5, 6), next(triples))
        self.assertRaises(StopIteration, lambda: next(triples))
