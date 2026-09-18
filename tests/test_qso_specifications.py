import unittest
from unittest import mock

import numpy as np

from tools.QSO_specifications import QSOSpecs


class RedshiftDistributionTests(unittest.TestCase):
    def test_distribution_is_reproducible_with_generator(self):
        specs = {"z_min": 0.2, "z_max": 2.0, "n_QSO": 5}
        first = QSOSpecs.redshift_distribution(specs, rng=np.random.default_rng(9))
        second = QSOSpecs.redshift_distribution(specs, rng=np.random.default_rng(9))
        np.testing.assert_array_equal(first, second)

    def test_random_redshifts_are_sorted_and_seeded(self):
        specs = {"z_min": 0.2, "z_max": 2.0, "n_QSO": 20}
        first = QSOSpecs.redshift_distribution(
            specs, rng=np.random.default_rng(9), mode="random"
        )
        second = QSOSpecs.redshift_distribution(
            specs, rng=np.random.default_rng(9), mode="random"
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.diff(first) >= 0))

    def test_asimov_redshifts_are_midpoint_quantiles_and_ignore_rng(self):
        specs = {"z_min": 0.0, "z_max": 1.0, "n_QSO": 4}
        expected = np.array([0.125, 0.375, 0.625, 0.875])
        with mock.patch(
            "tools.QSO_specifications.np.random.default_rng",
            side_effect=AssertionError("Asimov must not construct a generator"),
        ):
            first = QSOSpecs.redshift_distribution(specs, mode="asimov")
            second = QSOSpecs.redshift_distribution(specs, mode="asimov")
        np.testing.assert_allclose(first, expected)
        np.testing.assert_array_equal(first, second)

    def test_invalid_redshift_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "random.*asimov"):
            QSOSpecs.redshift_distribution(
                {"redshifts": [0.8, 0.2]},
                mode="validated",
            )

    def test_explicit_redshifts_override_uniform_placeholder(self):
        input_redshifts = np.array([1.2, 0.1, 0.3])
        specs = {"redshifts": input_redshifts}
        result = QSOSpecs.redshift_distribution(specs)
        np.testing.assert_array_equal(
            result, np.array([0.1, 0.3, 1.2])
        )
        np.testing.assert_array_equal(input_redshifts, np.array([1.2, 0.1, 0.3]))
        self.assertFalse(np.shares_memory(result, input_redshifts))

    def test_invalid_specification_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "z_min"):
            QSOSpecs.redshift_distribution(
                {"z_min": 1.0, "z_max": 0.2, "n_QSO": 4}
            )


if __name__ == "__main__":
    unittest.main()
