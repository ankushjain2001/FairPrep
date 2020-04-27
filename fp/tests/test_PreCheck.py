import os
import pickle
import unittest
from datetime import datetime
from fp.traindata_samplers import CompleteData, BalancedExamplesSampler
from fp.missingvalue_handlers import CompleteCaseAnalysis, ModeImputer, DataWigSimpleImputer
from fp.scalers import NoScaler, NamedStandardScaler, NamedMinMaxScaler
from fp.learners import NonTunedLogisticRegression, NonTunedDecisionTree, DecisionTree, LogisticRegression, AdversarialDebiasing
from fp.post_processors import NoPostProcessing, RejectOptionPostProcessing, EqualOddsPostProcessing, CalibratedEqualOddsPostProcessing
from fp.pre_processors import NoPreProcessing, Reweighing, DIRemover, LFR
from fp.experiments import BinaryClassificationExperiment

class test_pre_check(unittest.TestCase):
    def test_unittest(self):
        num1 = 3
        num2 = 10
        self.assertEqual(num1+num2, 13)
        
if __name__ == '__main__':
    unittest.main()
