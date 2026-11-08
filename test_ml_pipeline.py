"""Tests for HW2_ML_Pipeline_Consolidated.py.

Run: python -m unittest -v test_ml_pipeline.py
Optional real-data verification (set the folder holding the original CSV):
    PowerShell: $env:HW2_DATA_DIR = '.\\data'
    python -m unittest -v test_ml_pipeline.py

The real-data test checks the documented 100-tree/sqrt OOB accuracy; it does
not claim to reproduce every reported result. No original CSV is included here.
"""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

SCRIPT = Path(__file__).with_name('HW2_ML_Pipeline_Consolidated.py')
spec = importlib.util.spec_from_file_location('hw2_pipeline', SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def make_data(n=60):
    """Small artificial dataset with 608 numeric features and both classes."""
    rng = np.random.default_rng(123)
    cols = list(pipeline.RUNTIME_GENES) + [f'gene_{i:03d}' for i in range(602)]
    x = rng.normal(size=(n, 608))
    y = np.array([0, 1] * (n // 2))
    x[:, 0] += y * 3
    df = pd.DataFrame(x, columns=cols)
    df[pipeline.LABEL] = y
    return df


class PipelineUnitTests(unittest.TestCase):
    def test_split_has_no_overlap_and_preserves_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            original = make_data()
            original.to_csv(folder / 'Original training DB e1 cluster.csv', index=False)
            training, verification = pipeline.split_original(folder)
            self.assertEqual(len(training), len(original) - 2)
            self.assertEqual(len(verification), 2)
            self.assertEqual(set(verification.Label), {0, 1})
            self.assertTrue(training.index.intersection(verification.index).empty)
            self.assertTrue((folder / 'training_db.csv').is_file())
            self.assertTrue((folder / 'verification_db.csv').is_file())

    def test_metrics_match_confusion_matrix(self):
        actual = [0, 0, 1, 1, 1]
        predicted = [0, 1, 1, 0, 1]
        result = pipeline.metrics(actual, predicted)
        self.assertEqual((result['TN'], result['FP'], result['FN'], result['TP']), (1, 1, 1, 2))
        self.assertAlmostEqual(result['accuracy'], 3 / 5)
        self.assertAlmostEqual(result['e1_F1'], 2 / 3)

    def test_folds_cover_every_row_once_and_preserve_classes(self):
        y = make_data()[pipeline.LABEL]
        folds = pipeline.stratified_folds(y)
        self.assertEqual(len(folds), 3)
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(len(y))))
        for fold in folds:
            self.assertEqual(set(y.iloc[fold]), {0, 1})

    def test_validate_rejects_missing_values(self):
        training = make_data()
        verification = make_data(10)
        training.loc[0, pipeline.RUNTIME_GENES[0]] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing values'):
            pipeline.validate(training, verification)

    def test_end_to_end_outputs_on_synthetic_data(self):
        # Speed up the smoke test without changing the published pipeline script.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_data().to_csv(folder / 'Original training DB e1 cluster.csv', index=False)
            original_forest = pipeline.forest
            def fast_forest(n_estimators=100, max_features='sqrt', oob_score=False):
                return original_forest(n_estimators=40, max_features=max_features, oob_score=oob_score)
            with patch.object(pipeline, 'forest', side_effect=fast_forest), \
                 patch.object(pipeline, 'TREE_COUNTS', (40,)), \
                 patch.object(pipeline, 'FEATURE_SETTINGS', ('sqrt',)):
                train, verify = pipeline.split_original(folder)
                X, y = pipeline.validate(train, verify)
                oob = pipeline.oob_experiment(X, y, folder)
                folds = pipeline.stratified_folds(y)
                cv = pipeline.cross_validation(X, y, folds, folder)
                importance = pipeline.rank_features(X, y, folder)
                reduced = pipeline.reduced_features(X, y, folds, folder)
                runtime = pipeline.runtime_verify(train, verify, folder)
            self.assertEqual(len(oob), 1)
            self.assertEqual(len(cv), 3)
            self.assertEqual(len(importance), 608)
            self.assertEqual(len(reduced), 12)
            self.assertEqual(len(runtime), 2)
            self.assertTrue(runtime.Correct.isin([True, False]).all())
            for name in ('HW2_OOB_results.csv', 'HW2_CV_fold_results.csv',
                         'HW2_RF_Feature_Ranking.csv', 'HW2_Reduced_Feature_Pooled.csv',
                         'HW2_Runtime_Verification.csv', 'HW2_RF_Top10_Features.png'):
                self.assertTrue((folder / name).is_file(), name)


class OriginalDatasetReproducibility(unittest.TestCase):
    @unittest.skipUnless(os.getenv('HW2_DATA_DIR'), 'Set HW2_DATA_DIR to run the real-data test')
    def test_reported_oob_accuracy(self):
        folder = Path(os.environ['HW2_DATA_DIR'])
        source = folder / 'Original training DB e1 cluster.csv'
        self.assertTrue(source.is_file(), f'Missing dataset: {source}')
        df = pd.read_csv(source)
        self.assertEqual(df.shape, (871, 609))
        self.assertEqual(df.Label.value_counts().to_dict(), {0: 572, 1: 299})
        # Reproduce the deterministic withholding without writing into the data folder.
        a = df[df.Label == 1].sample(n=1, random_state=pipeline.SEED)
        b = df[df.Label == 0].sample(n=1, random_state=pipeline.SEED)
        train = df.drop(index=pd.concat([a, b]).index)
        X, y = pipeline.validate(train, pd.concat([a, b]))
        rf = pipeline.forest(100, 'sqrt', oob_score=True)
        rf.fit(X, y)
        pred = rf.classes_[rf.oob_decision_function_.argmax(axis=1)]
        result = pipeline.metrics(y, pred)
        self.assertEqual(sum(confusion_matrix(y, pred).ravel()), 869)
        # Reported 0.99655 corresponds to 3 errors in 869 rows.
        self.assertEqual(result['FP'] + result['FN'], 3)
        self.assertAlmostEqual(result['accuracy'], 0.99655, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
