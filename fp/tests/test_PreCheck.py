import os
import pickle
import unittest
from datetime import datetime
from freezegun import freeze_time

from fp.traindata_samplers import CompleteData, BalancedExamplesSampler
from fp.missingvalue_handlers import CompleteCaseAnalysis, ModeImputer, DataWigSimpleImputer
from fp.scalers import NoScaler, NamedStandardScaler, NamedMinMaxScaler
from fp.learners import NonTunedLogisticRegression, NonTunedDecisionTree, DecisionTree, LogisticRegression, AdversarialDebiasing
from fp.post_processors import NoPostProcessing, RejectOptionPostProcessing, EqualOddsPostProcessing, CalibratedEqualOddsPostProcessing
from fp.pre_processors import NoPreProcessing, Reweighing, DIRemover, LFR
from fp.experiments import BinaryClassificationExperiment

@freeze_time('2020-01-01 00:00:00.000000')
class test_pre_check(unittest.TestCase):
    def test_unittest(self):
        num1 = 3
        num2 = 10
        self.assertEqual(num1+num2, 13)

    def test_freezegun(self):
        self.assertEqual(datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3], '2020-01-01_00-00-00-000')
        
if __name__ == '__main__':
    unittest.main()