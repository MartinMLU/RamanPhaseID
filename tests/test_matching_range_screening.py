import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "raman_phase_id", PROJECT_ROOT / "RamanPhaseID_0p99beta.py"
)
APP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(APP)


class MatchingRangeScreeningTests(unittest.TestCase):
    @staticmethod
    def _full_support_normalize(row):
        row = np.asarray(row, dtype=np.float32)
        row = row - np.min(row)
        peak = np.max(row)
        return row / peak if peak > 0.0 else row

    def _rank(self, rows, query, support, meta):
        rows = np.asarray(rows, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "X.float32"
            matrix = np.memmap(path, mode="w+", dtype=np.float32, shape=rows.shape)
            matrix[:] = rows
            matrix.flush()
            ranked = APP._topk_cosine_subset(
                np.asarray(query, dtype=np.float32),
                matrix,
                meta,
                np.arange(rows.shape[0], dtype=np.int32),
                rows.shape[0],
                support_mask=np.asarray(support, dtype=bool),
            )
            del matrix
        return ranked

    def test_outside_range_intensity_cannot_change_candidate_order(self):
        query = self._full_support_normalize([2.0, 3.0, 2.0, 2.0])
        support = [True, True, False, False]
        meta = [
            {"start_idx": 0, "end_idx": 3, "l2": 1.0e9},
            {"start_idx": 0, "end_idx": 3, "l2": 0.01},
        ]

        without_outside_peak = self._rank(
            [
                self._full_support_normalize([2.0, 3.0, 2.0, 2.0]),
                self._full_support_normalize([3.0, 2.0, 2.0, 2.0]),
            ],
            query,
            support,
            meta,
        )
        with_outside_peak = self._rank(
            [
                # Only the raw value outside support changed.  Full-support
                # normalisation propagates that new minimum into columns 0:2.
                self._full_support_normalize([2.0, 3.0, -100.0, 2.0]),
                self._full_support_normalize([3.0, 2.0, 2.0, 2.0]),
            ],
            query,
            support,
            meta,
        )

        self.assertEqual(without_outside_peak, [0, 1])
        self.assertEqual(with_outside_peak, without_outside_peak)

    def test_query_and_database_norms_use_each_references_native_coverage(self):
        # Row 0 is an exact match on its advertised two-point coverage.  Large
        # stale values after end_idx must not enter either its dot product or
        # its norm, and the query norm must use that same two-point support.
        query = [1.0, 2.0, 10.0, 10.0]
        support = [True, True, True, True]
        rows = [
            [1.0, 2.0, 1.0e6, 1.0e6],
            [1.0, 2.0, 10.0, 0.0],
        ]
        meta = [
            {"start_idx": 0, "end_idx": 1, "l2": 1.0e12},
            {"start_idx": 0, "end_idx": 3, "l2": 1.0},
        ]

        self.assertEqual(self._rank(rows, query, support, meta), [0, 1])

    def test_refinement_scores_ignore_full_support_affine_normalization(self):
        axis = np.arange(100, dtype=float)
        query = (
            np.exp(-0.5 * ((axis - 25.0) / 2.0) ** 2)
            + 0.7 * np.exp(-0.5 * ((axis - 52.0) / 3.0) ** 2)
        )
        candidate = query + 0.35 * np.exp(-0.5 * ((axis - 76.0) / 2.5) ** 2)
        candidate_after_outside_extremum = (0.025 * candidate) + 0.91
        support = (axis >= 10.0) & (axis <= 90.0)

        shape_a, _ = APP._best_aligned_score(
            query, candidate, support, 0, axis.size - 1, max_shift=0
        )
        shape_b, _ = APP._best_aligned_score(
            query,
            candidate_after_outside_extremum,
            support,
            0,
            axis.size - 1,
            max_shift=0,
        )
        pcs_a = APP._peak_consistency_score(query, candidate, support)[0]
        pcs_b = APP._peak_consistency_score(
            query, candidate_after_outside_extremum, support
        )[0]

        self.assertAlmostEqual(shape_a, shape_b, places=12)
        self.assertAlmostEqual(pcs_a, pcs_b, places=12)

    def test_refinement_gradient_does_not_read_outside_matching_range(self):
        query = np.array([7.0, 9.0, 0.0, 1.0, 0.2, 0.8, 0.1, 0.6, 8.0, 6.0])
        candidate = query.copy()
        changed_outside = candidate.copy()
        changed_outside[:2] = [-1.0e6, 1.0e6]
        changed_outside[8:] = [1.0e6, -1.0e6]
        support = np.zeros(query.size, dtype=bool)
        support[2:8] = True

        score_a, _ = APP._best_aligned_score(
            query, candidate, support, 0, query.size - 1, max_shift=0
        )
        score_b, _ = APP._best_aligned_score(
            query, changed_outside, support, 0, query.size - 1, max_shift=0
        )

        self.assertAlmostEqual(score_a, score_b, places=12)


if __name__ == "__main__":
    unittest.main()
