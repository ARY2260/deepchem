# flake8: noqa

"""
Test for DMPNN Featurizer class
"""

from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.molecule_featurizers.dmpnn_featurizer import DMPNNFeaturizer, GraphConvConstants
from rdkit import Chem
import pytest
import numpy as np


def test_default_featurizer():
  """
    Test for featurization of 2 smiles using `DMPNNFeaturizer` class with no input parameters
    """
  smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
  featurizer = DMPNNFeaturizer()
  graph_feat = featurizer.featurize(smiles)
  assert len(graph_feat) == 2

  # assert "C1=CC=CN=C1"
  assert graph_feat[0].num_nodes == 6
  assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
  assert graph_feat[0].node_features.shape == (6, GraphConvConstants.ATOM_FDIM)
  assert graph_feat[0].node_features_zero_padded.shape == (
      6 + 1, GraphConvConstants.ATOM_FDIM)
  assert graph_feat[0].num_edges == 12
  assert graph_feat[0].concatenated_features_zero_padded.shape == (
      12 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
  assert len(graph_feat[0].mapping) == 12 + 1

  # # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
  assert graph_feat[1].num_nodes == 22
  assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
  assert graph_feat[1].node_features.shape == (22, GraphConvConstants.ATOM_FDIM)
  assert graph_feat[1].node_features_zero_padded.shape == (
      22 + 1, GraphConvConstants.ATOM_FDIM)
  assert graph_feat[1].num_edges == 44
  assert graph_feat[1].concatenated_features_zero_padded.shape == (
      44 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
  assert len(graph_feat[1].mapping) == 44 + 1


def test_featurizer_with_adding_hs():
  """
    Test for featurization of 2 smiles using `DMPNNFeaturizer` class with `is_adding_hs` set to `True`
    """
  smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
  featurizer = DMPNNFeaturizer(is_adding_hs=True)
  graph_feat = featurizer.featurize(smiles)
  assert len(graph_feat) == 2

  # assert "C1=CC=CN=C1"
  assert graph_feat[0].num_nodes == 11
  assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
  assert graph_feat[0].node_features.shape == (11, GraphConvConstants.ATOM_FDIM)
  assert graph_feat[0].node_features_zero_padded.shape == (
      11 + 1, GraphConvConstants.ATOM_FDIM)
  assert graph_feat[0].num_edges == 22
  assert graph_feat[0].concatenated_features_zero_padded.shape == (
      22 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
  assert len(graph_feat[0].mapping) == 22 + 1

  # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
  assert graph_feat[1].num_nodes == 49
  assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
  assert graph_feat[1].node_features.shape == (49, GraphConvConstants.ATOM_FDIM)
  assert graph_feat[1].node_features_zero_padded.shape == (
      49 + 1, GraphConvConstants.ATOM_FDIM)
  assert graph_feat[1].num_edges == 98
  assert graph_feat[1].concatenated_features_zero_padded.shape == (
      98 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
  assert len(graph_feat[1].mapping) == 98 + 1


class DummyTestClass(MolecularFeaturizer):
  """
    Dummy test class derived from MolecularFeaturizer where `use_original_atom_ranks` parameter is not initialised
    """

  def __init__(self):
    pass

  def _featurize(self, datapoint, **kwargs):
    """
        returns mapping of atomic number and atom ranks as feature vector (only for testing purposes)
        """
    if isinstance(datapoint, Chem.rdchem.Mol):
      atoms_ranks = []
      for atom in datapoint.GetAtoms():
        atoms_ranks.append((atom.GetAtomicNum(), atom.GetIdx()))
      return atoms_ranks


class DummyTestClass2(MolecularFeaturizer):
  """
    Dummy test class derived from MolecularFeaturizer where `use_original_atom_ranks` parameter is initialised
    """

  def __init__(self, use_original_atom_ranks=False):
    self.use_original_atom_ranks = use_original_atom_ranks

  def _featurize(self, datapoint, **kwargs):
    """
        returns mapping of atomic number and atom ranks as feature vector (only for testing purposes)
        """
    if isinstance(datapoint, Chem.rdchem.Mol):
      atoms_ranks = []
      for atom in datapoint.GetAtoms():
        atoms_ranks.append((atom.GetAtomicNum(), atom.GetIdx()))
      return atoms_ranks


def test_use_original_atom_ranks():
  """
    Test for `use_original_atom_ranks` boolean condition added to `MolecularFeaturizer` base class
    """
  from rdkit.Chem import rdmolfiles
  from rdkit.Chem import rdmolops
  smile = "C1=CC=CN=C1"
  mol = Chem.MolFromSmiles(smile)

  original_atom_ranks = []
  for atom in mol.GetAtoms():
    original_atom_ranks.append(
        (atom.GetAtomicNum(),
         atom.GetIdx()))  # mapping of atomic number and original atom ranking

  new_order = rdmolfiles.CanonicalRankAtoms(mol)
  mol = rdmolops.RenumberAtoms(mol, new_order)
  canonical_atom_ranks = []
  for atom in mol.GetAtoms():
    canonical_atom_ranks.append(
        (atom.GetAtomicNum(),
         atom.GetIdx()))  # mapping of atomic number and canonical atom ranking

  # test without use_original_atom_ranks being initialised
  featurizer = DummyTestClass()
  datapoint_atom_ranks = featurizer.featurize(
      smile)  # should be canonical mapping
  print(datapoint_atom_ranks)
  print(np.asarray([canonical_atom_ranks]))
  assert (datapoint_atom_ranks == np.asarray([canonical_atom_ranks])).all()

  # test with use_original_atom_ranks = False
  featurizer = DummyTestClass2(use_original_atom_ranks=False)
  datapoint_atom_ranks = featurizer.featurize(
      smile)  # should be canonical mapping
  print(datapoint_atom_ranks)
  print(np.asarray([canonical_atom_ranks]))
  assert (datapoint_atom_ranks == np.asarray([canonical_atom_ranks])).all()

  # test with use_original_atom_ranks = True
  featurizer = DummyTestClass2(use_original_atom_ranks=True)
  datapoint_atom_ranks = featurizer.featurize(
      smile)  # should be canonical mapping
  print(datapoint_atom_ranks)
  print(np.asarray([original_atom_ranks]))
  assert (datapoint_atom_ranks == np.asarray([original_atom_ranks])).all()
