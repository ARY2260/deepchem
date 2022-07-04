"""
Test basic molecular features.
"""
import numpy as np
import unittest

from deepchem.feat import RDKitDescriptors, RDKit2DFeaturizer


class TestRDKitDescriptors(unittest.TestCase):
  """
  Test RDKitDescriptors.
  """

  def setUp(self):
    """
    Set up tests.
    """
    from rdkit import Chem
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(smiles)
    self.featurizer = RDKitDescriptors()

  def test_rdkit_descriptors(self):
    """
    Test simple descriptors.
    """
    featurizer = RDKitDescriptors()
    descriptors = featurizer([self.mol])
    assert descriptors.shape == (1, len(featurizer.descriptors))
    assert np.allclose(
        descriptors[0, featurizer.descriptors.index('ExactMolWt')],
        180,
        atol=0.1)

  def test_rdkit_descriptors_on_smiles(self):
    """
    Test invocation on raw smiles.
    """
    featurizer = RDKitDescriptors()
    descriptors = featurizer('CC(=O)OC1=CC=CC=C1C(=O)O')
    assert descriptors.shape == (1, len(featurizer.descriptors))
    assert np.allclose(
        descriptors[0, featurizer.descriptors.index('ExactMolWt')],
        180,
        atol=0.1)

  def test_rdkit_descriptors_with_use_fragment(self):
    """
    Test with use_fragment
    """
    from rdkit.Chem import Descriptors
    featurizer = RDKitDescriptors(use_fragment=False)
    descriptors = featurizer(self.mol)
    assert descriptors.shape == (1, len(featurizer.descriptors))
    all_descriptors = Descriptors.descList
    assert len(featurizer.descriptors) < len(all_descriptors)
    assert np.allclose(
        descriptors[0, featurizer.descriptors.index('ExactMolWt')],
        180,
        atol=0.1)

class TestRDKit2DFeaturizer(unittest.TestCase):
  """
  Test for `RDKit2DFeaturizer` class.
  """

  def setUp(self):
    """
    Set up tests.
    """
    from rdkit import Chem
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(smiles)
  
  def test_default_featurizer(self):
    """
    Test for featurization of given smile using `RDKit2DFeaturizer` class with no input parameters.
    """
    featurizer = RDKit2DFeaturizer()
    features = featurizer.featurize(self.mol)
    assert features.shape == (1, 200)

  def test_featurizer_normalised(self):
    """
    Test for featurization of given smile using `RDKit2DFeaturizer` class with normalization
    """
    featurizer = RDKit2DFeaturizer(is_normalized=True)
    features = featurizer.featurize(self.mol)
    assert features.shape == (1, 200)
    assert len(np.where(features>1.0)[0]) == 0
