import os
import deepchem as dc
import unittest
from deepchem.data import Dataset, DiskDataset
from deepchem.feat.molecule_featurizers.dmpnn_featurizer import DMPNNFeaturizer, GraphConvConstants

class TestDMPNNFeaturizerGenDataset(unittest.TestCase):
  def setUp(self):
    base_dir = os.getcwd()
    self.data_dir = os.path.join(base_dir, "trial/dmpnn_test")
  
  def test_freesolv(self):
    tasks, datasets, trans = dc.molnet.load_freesolv(featurizer=DMPNNFeaturizer(), splitter = None, transformers=[], data_dir=self.data_dir, save_dir=self.data_dir)
    assert isinstance(datasets[0], DiskDataset)
    for (xi, yi, wi, idi) in datasets[0].itersamples():
      print(xi)
      print()
      print(yi)
      print()
      print(wi)
      print()
      print(idi)
      break
  
  def test_freesolv_morgan(self):
    tasks, datasets, trans = dc.molnet.load_freesolv(featurizer=DMPNNFeaturizer(is_adding_hs=True, features_generators=['morgan']), splitter = None, transformers=[], data_dir=self.data_dir, save_dir=self.data_dir)
    assert isinstance(datasets[0], DiskDataset)
    for (xi, yi, wi, idi) in datasets[0].itersamples():
      print(xi)
      print()
      print(yi)
      print()
      print(wi)
      print()
      print(idi)
      break