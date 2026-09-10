import unittest

from prop_trader.shots_sweep import rank


class RankTests(unittest.TestCase):
    def test_higher_direction_accuracy_wins(self):
        results = [dict(shots=4, mae=.01, da5=.55), dict(shots=8, mae=.01, da5=.62)]
        self.assertEqual(rank(results)[0]['shots'], 8)

    def test_ties_broken_by_lower_mae(self):
        results = [dict(shots=4, mae=.02, da5=.6), dict(shots=8, mae=.01, da5=.6)]
        self.assertEqual(rank(results)[0]['shots'], 8)

    def test_none_direction_accuracy_ranks_last(self):
        results = [dict(shots=4, mae=.01, da5=None), dict(shots=8, mae=.01, da5=.5)]
        self.assertEqual(rank(results)[0]['shots'], 8)

    def test_full_ordering_is_stable_and_complete(self):
        results = [dict(shots=4, mae=.02, da5=.5), dict(shots=8, mae=.01, da5=.6), dict(shots=16, mae=.03, da5=.55)]
        self.assertEqual([r['shots'] for r in rank(results)], [8, 16, 4])


if __name__ == '__main__':
    unittest.main()
