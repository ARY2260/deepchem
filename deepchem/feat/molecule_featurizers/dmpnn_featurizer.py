# flake8: noqa

import numpy as np
from rdkit import Chem
from typing import List, Tuple, Union, Dict, Set, Sequence
import deepchem as dc
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol

from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer

from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot

from deepchem.feat.graph_features import bond_features as b_Feats


class GraphConvConstants(object):
  """
  A class for holding featurization parameters.
  """

  MAX_ATOMIC_NUM = 100
  ATOM_FEATURES: Dict[str, List[int]] = {
      'atomic_num': list(range(MAX_ATOMIC_NUM)),
      'degree': [0, 1, 2, 3, 4, 5],
      'formal_charge': [-1, -2, 1, 2, 0],
      'chiral_tag': [0, 1, 2, 3],
      'num_Hs': [0, 1, 2, 3, 4]
  }
  ATOM_FEATURES_HYBRIDIZATION: List[str] = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
  """Dimension of atom feature vector"""
  ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + len(
      ATOM_FEATURES_HYBRIDIZATION) + 1 + 2
  # len(choices) +1 and len(ATOM_FEATURES_HYBRIDIZATION) +1 to include room for unknown set
  # + 2 at end for is_in_aromatic and mass
  BOND_FDIM = 14


def get_atomic_num_one_hot(atom: RDKitAtom,
                           allowable_set: List[int],
                           include_unknown_set: bool = True) -> List[float]:
  """Get a one-hot feature about atomic number of the given atom.

  Parameters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  allowable_set: List[int]
    The range of atomic numbers to consider.
  include_unknown_set: bool, default False
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[float]
    A one-hot vector of atomic number of the given atom.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  return one_hot_encode(atom.GetAtomicNum() - 1, allowable_set,
                        include_unknown_set)


def get_atom_chiral_tag_one_hot(
    atom: RDKitAtom,
    allowable_set: List[int],
    include_unknown_set: bool = True) -> List[float]:
  """Get a one-hot feature about chirality of the given atom.

  Parameters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  allowable_set: List[int]
    The list of chirality tags to consider.
  include_unknown_set: bool, default False
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[float]
    A one-hot vector of chirality of the given atom.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  return one_hot_encode(atom.GetChiralTag(), allowable_set, include_unknown_set)


def get_atom_mass(atom: RDKitAtom) -> List[float]:
  """Get vector feature containing downscaled mass of the given atom.

  Parameters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object

  Returns
  -------
  List[float]
    A vector of downscaled mass of the given atom.
  """
  return [atom.GetMass() * 0.01]


def atom_features(
    atom: Chem.rdchem.Atom,
    functional_groups: List[int] = None,
    only_atom_num: bool = False) -> Sequence[Union[bool, int, float]]:
  """Helper method used to compute atom feature vector.

  Deepchem already contains an atom_features function, however we are defining a new one here due to the need to handle features specific to DMPNN.

  Parameters
  ----------
  atom: RDKit.Chem.rdchem.Atom
    Atom to compute features on.
  functional_groups: List[int]
    A k-hot vector indicating the functional groups the atom belongs to.
    Default value is None
  only_atom_num: bool
    Toggle to build a feature vector for an atom containing only the atom number information.

  Returns
  -------
  features: Sequence[Union[bool, int, float]]
    A list of atom features.

  Examples
  --------
  >>> from rdkit import Chem
  >>> mol = Chem.MolFromSmiles('C')
  >>> atom = mol.GetAtoms()[0]
  >>> features = dc.feat.molecule_featurizers.dmpnn_featurizer.atom_features(atom)
  >>> type(features)
  <class 'list'>
  >>> len(features)
  133
  """

  if atom is None:
    features: Sequence[Union[bool, int,
                             float]] = [0] * GraphConvConstants.ATOM_FDIM

  elif only_atom_num:
    features = []
    features += get_atomic_num_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
    features += [0] * (
        GraphConvConstants.ATOM_FDIM - GraphConvConstants.MAX_ATOMIC_NUM - 1
    )  # set other features to zero

  else:
    features = []
    features += get_atomic_num_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
    features += get_atom_total_degree_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['degree'])
    features += get_atom_formal_charge_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['formal_charge'])
    features += get_atom_chiral_tag_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['chiral_tag'])
    features += get_atom_total_num_Hs_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['num_Hs'])
    features += get_atom_hybridization_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES_HYBRIDIZATION, True)
    features += get_atom_is_in_aromatic_one_hot(atom)
    features = [int(feature) for feature in features]
    features += get_atom_mass(atom)

    if functional_groups is not None:
      features += functional_groups
  return features


def bond_features(bond: Chem.rdchem.Bond) -> Sequence[Union[bool, int, float]]:
  """wrapper function for bond_features() already available in deepchem, used to compute bond feature vector.

  Parameters
  ----------
  bond: rdkit.Chem.rdchem.Bond
    Bond to compute features on.

  Returns
  -------
  features: Sequence[Union[bool, int, float]]
    A list of bond features.

  Examples
  --------
  >>> from rdkit import Chem
  >>> mol = Chem.MolFromSmiles('CC')
  >>> bond = mol.GetBondWithIdx(0)
  >>> b_features = dc.feat.molecule_featurizers.dmpnn_featurizer.bond_features(bond)
  >>> type(b_features)
  <class 'list'>
  >>> len(b_features)
  14
  """
  if bond is None:
    b_features: Sequence[Union[
        bool, int, float]] = [1] + [0] * (GraphConvConstants.BOND_FDIM - 1)

  else:
    b_features = [0] + b_Feats(bond, use_extended_chirality=True)
  return b_features


def map_reac_to_prod(
    mol_reac: Chem.Mol,
    mol_prod: Chem.Mol) -> Tuple[Dict[int, int], List[int], List[int]]:
  """
  Function to build a dictionary of mapping atom indices in the reactants to the products.

  Parameters
  ----------
  mol_reac: Chem.Mol
  An RDKit molecule of the reactants.

  mol_prod: Chem.Mol
  An RDKit molecule of the products.

  Returns
  -------
  mappings: Tuple[Dict[int,int],List[int],List[int]]
  A tuple containing a dictionary of corresponding reactant and product atom indices,
  list of atom ids of product not part of the mapping and
  list of atom ids of reactant not part of the mapping
  """
  only_prod_ids: List[int] = []
  prod_map_to_id: Dict[int, int] = {}
  mapnos_reac: Set[int] = set(
      [atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()])
  for atom in mol_prod.GetAtoms():
    mapno = atom.GetAtomMapNum()
    if (mapno > 0):
      prod_map_to_id[mapno] = atom.GetIdx()
      if (mapno not in mapnos_reac):
        only_prod_ids.append(atom.GetIdx())
    else:
      only_prod_ids.append(atom.GetIdx())
  only_reac_ids: List[int] = []
  reac_id_to_prod_id: Dict[int, int] = {}
  for atom in mol_reac.GetAtoms():
    mapno = atom.GetAtomMapNum()
    if (mapno > 0):
      try:
        reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
      except KeyError:
        only_reac_ids.append(atom.GetIdx())
    else:
      only_reac_ids.append(atom.GetIdx())
  mappings: Tuple[Dict[int, int], List[int],
                  List[int]] = (reac_id_to_prod_id, only_prod_ids,
                                only_reac_ids)
  return mappings


class DMPNNFeaturizer(MolecularFeaturizer):
  """
  This class is a featurizer for Directed Message Passing Neural Network (D-MPNN) implementation
  """

  def __init__(self,
               features_generator,
               is_adding_hs: bool = False,
               features_scaling: bool = False,
               atom_descriptor_scaling: bool = False,
               bond_feature_scaling: bool = False) -> None:

    self.features_generator = features_generator
    self.is_adding_hs = is_adding_hs
    self.features_scaling = features_scaling
    self.atom_descriptor_scaling = atom_descriptor_scaling
    self.bond_feature_scaling = bond_feature_scaling
    self.atom_descriptors = None
    self.phase_features = None

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:

    if isinstance(datapoint, Chem.rdchem.Mol):
      if self.is_adding_hs:
        datapoint = Chem.AddHs(datapoint)
    else:
      raise ValueError(
          "Feature field should contain smiles for DMPNN featurizer!")

    num_atoms: int  # number of atoms
    num_bonds: int = 1  # number of bonds

    # mapping from atom index to atom features | initial input is a zero padding
    f_atoms: np.ndarray = np.asarray([[0] * GraphConvConstants.ATOM_FDIM])

    # mapping from bond index to concat(in_atom, bond) features | initial input is a zero padding
    f_ini_atoms_bonds: np.ndarray = np.asarray(
        [[0] * (GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)])

    # mapping from bond index to the index of the atom the bond is coming from
    bond_to_ini_atom: List[int] = [0]

    # mapping from bond index to the index of the reverse bond
    b2revb: List[int] = [0]

    # get atom features
    np.concatenate(
        (f_atoms,
         np.asarray([atom_features(atom) for atom in datapoint.GetAtoms()])),
        axis=0)
    num_atoms = len(f_atoms) - 1

    mapping: np.ndarray = np.zeros([num_atoms + 1, num_atoms + 1], dtype=int)

    for a1 in range(1, num_atoms + 1):  # get matrix mapping of bonds
      for a2 in range(a1 + 1, num_atoms + 1):

        bond = datapoint.GetBondBetweenAtoms(a1 - 1, a2 - 1)

        if bond is None:
          continue

        # get bond features
        f_bond: np.ndarray = np.asarray(bond_features(bond))

        np.append(f_ini_atoms_bonds, f_atoms[a1].extend(f_bond))
        np.append(f_ini_atoms_bonds, f_atoms[a2].extend(f_bond))

        b1 = num_bonds
        b2 = num_bonds + 1
        mapping[a1][a2] = b1
        mapping[a2][a1] = b2

        bond_to_ini_atom.append(a1)
        bond_to_ini_atom.append(a2)
        num_bonds += 2

        b2revb.append(b2)
        b2revb.append(b1)

    # get mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond
    mapping = mapping.transpose()[bond_to_ini_atom]

    # replace the reverse bonds with zeros
    for count, i in enumerate(b2revb):
      mapping[count][np.where(mapping[count] == i)] = 0

    return GraphData(node_features=f_atoms,
                     edge_index=np.empty(0),
                     edge_features=f_ini_atoms_bonds,
                     global_features=None,
                     mapping=mapping)

  def _generate_global_features(self, datapoint: RDKitMol):
    # generate features and fix nans
    return NotImplementedError

  def _phase_features_generator(self, datapoint: RDKitMol):
    return NotImplementedError

  def _generate_atom_descriptors(self, datapoint: RDKitMol):
    return NotImplementedError
