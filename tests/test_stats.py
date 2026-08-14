import math
import unittest

import pandas as pd

from kinelearn_stimviz.stats import add_summary_statistics


class SummaryStatisticsTests(unittest.TestCase):
    def test_summary_uses_students_t_confidence_interval(self) -> None:
        values = pd.DataFrame({"group": ["a"] * 5, "value": [1, 2, 3, 4, 5]})

        summary = add_summary_statistics(values, group_cols=["group"])

        self.assertAlmostEqual(summary.loc[0, "mean"], 3.0)
        self.assertAlmostEqual(summary.loc[0, "sem"], math.sqrt(0.5))
        self.assertAlmostEqual(summary.loc[0, "ci_low"], 1.0367568385)
        self.assertAlmostEqual(summary.loc[0, "ci_high"], 4.9632431615)

    def test_single_observation_has_undefined_confidence_interval(self) -> None:
        values = pd.DataFrame({"group": ["a"], "value": [3.0]})

        summary = add_summary_statistics(values, group_cols=["group"])

        self.assertTrue(math.isnan(summary.loc[0, "ci_low"]))
        self.assertTrue(math.isnan(summary.loc[0, "ci_high"]))


if __name__ == "__main__":
    unittest.main()
