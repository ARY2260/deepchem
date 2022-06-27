# flake8: noqa

import numpy as np
from rdkit import Chem
from typing import List, Tuple, Union, Dict, Set, Sequence
import logging
import deepchem as dc
from deepchem.utils.typing import RDKitAtom, RDKitMol

from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.molecule_featurizers.circular_fingerprint import CircularFingerprint

from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot

from deepchem.feat.graph_features import bond_features as b_Feats

logger = logging.getLogger(__name__)


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
  FEATURE_GENERATORS = {"morgan": CircularFingerprint(radius = 2, size = 2048, sparse = False)}


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
  """This class is a featurizer for Directed Message Passing Neural Network (D-MPNN) implementation

  The default node(atom) and edge(bond) representations are based on
  `Analyzing Learned Molecular Representations for Property Prediction paper <https://arxiv.org/pdf/1904.01561.pdf>`_.

  The default node representation are constructed by concatenating the following values,
  and the feature length is 133.

  - Atomic num: A one-hot vector of this atom, in a range of first 100 atoms.
  - Degree: A one-hot vector of the degree (0-5) of this atom.
  - Formal charge: Integer electronic charge, -1, -2, 1, 2, 0.
  - Chirality: A one-hot vector of the chirality tag (0-3) of this atom.
  - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
  - Hybridization: A one-hot vector of "SP", "SP2", "SP3", "SP3D", "SP3D2".
  - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
  - Mass: Atomic mass * 0.01

  The default edge representation are constructed by concatenating the following values,
  and the feature length is 14.

  - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
  - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
  - Conjugated: A one-hot vector of whether this bond is conjugated or not.
  - Stereo: A one-hot vector of the stereo configuration (0-5) of a bond.

  If you want to know more details about features, please check the paper [1]_ and
  utilities in deepchem.utils.molecule_feature_utils.py.

  Examples
  --------
  >>> smiles = ["C1=CC=CN=C1", "C1CCC1"]
  >>> featurizer = DMPNNFeaturizer()
  >>> out = featurizer.featurize(smiles)
  >>> type(out[0])
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> out[0].num_node_features
  133
  >>> out[0].node_features.shape
  (6, 133)
  >>> out[0].node_features_zero_padded.shape
  (7, 133)
  >>> out[0].num_edges
  12
  >>> out[0].concatenated_features_zero_padded.shape
  (13, 147)
  >>> len(out[0].mapping)
  13

  References
  ----------
  .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
     Journal of computer-aided molecular design 30.8 (2016):595-608.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self,
               features_generator=None,
               is_adding_hs: bool = False,
               use_original_atom_ranks: bool = False,
               features_scaling: bool = False,
               atom_descriptor_scaling: bool = False,
               bond_feature_scaling: bool = False) -> None:
    """
    Parameters
    ----------
    features_generator:
      # TODO: Implement global feature generator
    is_adding_hs: bool, default False
      Whether to add Hs or not.
    use_original_atom_ranks: bool, default False
      Whether to use original atom mapping or canonical atom mapping
    features_scaling: bool, default False
      # TODO: feature normalization
    atom_descriptor_scaling: bool, default False
      # TODO: descriptor normalization
    bond_feature_scaling: bool, default False
      # TODO: bond feature normalization
    """
    self.features_generator = features_generator
    self.available_generators = GraphConvConstants.FEATURE_GENERATORS
    self.is_adding_hs = is_adding_hs
    self.use_original_atom_ranks = use_original_atom_ranks
    self.features_scaling = features_scaling
    self.atom_descriptor_scaling = atom_descriptor_scaling
    self.bond_feature_scaling = bond_feature_scaling
    self.atom_descriptors = None
    self.phase_features = None

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
    """Calculate molecule graph features from RDKit mol object.

    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit mol object.

    Returns
    -------
    graph: GraphData
      A molecule graph object with features:
      - node_features: Node feature matrix with shape [num_nodes, num_node_features]
      - edge_index: Graph connectivity in COO format with shape [2, num_edges]
      - global_features: None # TODO: to be implemented
      - mapping: Mapping from 'bond index' to array of indices (of the bonds incoming at the initial atom of the bond) with shape [num_nodes + 1, maximum incoming bonds]
      - node_features_zero_padded: Zero-padded node feature matrix with shape [num_nodes + 1, num_node_features]
      - concatenated_features_zero_padded: Zero-padded mapping from bond index to concatenated (incoming atom, bond) features with shape [num_edges + 1, num_node_features + num_bond_features]
    """
    if isinstance(datapoint, Chem.rdchem.Mol):
      if self.is_adding_hs:
        datapoint = Chem.AddHs(datapoint)
    else:
      raise ValueError(
          "Feature field should contain smiles for DMPNN featurizer!")

    global_features = self._generate_global_features(datapoint)

    num_atoms: int  # number of atoms
    num_bonds: int = 1  # number of bonds

    atom_fdim = GraphConvConstants.ATOM_FDIM
    bond_fdim = GraphConvConstants.BOND_FDIM
    concat_fdim = atom_fdim + bond_fdim

    # mapping from atom index to atom features | initial input is a zero padding
    f_atoms_zero_padded: np.ndarray = np.asarray([[0] * atom_fdim], dtype=float)

    # mapping from bond index to concat(in_atom, bond) features | initial input is a zero padding
    f_ini_atoms_bonds_zero_padded: np.ndarray = np.asarray(
        [[0] * (atom_fdim + bond_fdim)], dtype=float)

    # mapping from bond index to the index of the atom the bond is coming from
    bond_to_ini_atom: List[int] = [0]

    # mapping from bond index to the index of the reverse bond
    b2revb: List[int] = [0]

    # get atom features
    f_atoms: np.ndarray = np.asarray(
        [atom_features(atom) for atom in datapoint.GetAtoms()], dtype=float)

    f_atoms_zero_padded = np.concatenate((f_atoms_zero_padded, f_atoms), axis=0)
    num_atoms = len(f_atoms_zero_padded) - 1

    # mapping: np.ndarray = np.zeros([num_atoms + 1, num_atoms + 1], dtype=int)
    atom_to_incoming_bonds: List[List[int]] = [[] for i in range(num_atoms + 1)]

    for a1 in range(1, num_atoms + 1):  # get matrix mapping of bonds
      for a2 in range(a1 + 1, num_atoms + 1):
        bond = datapoint.GetBondBetweenAtoms(a1 - 1, a2 - 1)

        if bond is None:
          continue

        # get bond features
        f_bond: np.ndarray = np.asarray(bond_features(bond), dtype=float)

        _ = np.concatenate((f_atoms_zero_padded[a1], f_bond),
                           axis=0).reshape([1, concat_fdim])
        f_ini_atoms_bonds_zero_padded = np.concatenate(
            (f_ini_atoms_bonds_zero_padded, _), axis=0)

        _ = np.concatenate((f_atoms_zero_padded[a2], f_bond),
                           axis=0).reshape([1, concat_fdim])
        f_ini_atoms_bonds_zero_padded = np.concatenate(
            (f_ini_atoms_bonds_zero_padded, _), axis=0)

        b1 = num_bonds
        b2 = num_bonds + 1

        atom_to_incoming_bonds[a2].append(b1)  # b1 = a1 --> a2
        atom_to_incoming_bonds[a1].append(b2)  # b2 = a2 --> a1

        bond_to_ini_atom.append(a1)
        bond_to_ini_atom.append(a2)
        num_bonds += 2

        b2revb.append(b2)
        b2revb.append(b1)

    max_num_bonds = max(
        1,
        max(len(incoming_bonds) for incoming_bonds in atom_to_incoming_bonds))
    atom_to_incoming_bonds = [
        atom_to_incoming_bonds[a] + [0] *
        (max_num_bonds - len(atom_to_incoming_bonds[a]))
        for a in range(num_atoms + 1)
    ]

    # get mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond
    mapping = np.asarray(atom_to_incoming_bonds)[bond_to_ini_atom]

    # replace the reverse bonds with zeros
    for count, i in enumerate(b2revb):
      mapping[count][np.where(mapping[count] == i)] = 0

    # construct edge (bond) index
    src, dest = [], []
    for bond in datapoint.GetBonds():
      # add edge list considering a directed graph
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      src += [start, end]
      dest += [end, start]

    return GraphData(
        node_features=f_atoms,
        edge_index=np.asarray([src, dest], dtype=int),
        global_features=global_features,
        mapping=mapping,
        node_features_zero_padded=f_atoms_zero_padded,
        concatenated_features_zero_padded=f_ini_atoms_bonds_zero_padded)

  def _generate_global_features(self, datapoint: RDKitMol):
    # TODO: generate features and fix nans
    global_features = []
    generators_list = self.available_generators.keys()
    for generator in self.features_generator:
      if generator in generators_list:
        global_featurizer = self.available_generators[generator]
        if datapoint.GetNumHeavyAtoms() > 0:
            global_features.extend(global_featurizer.featurize(datapoint)[0])
        # for H2
        elif datapoint.GetNumHeavyAtoms() == 0:
            # not all features are equally long, so used methane as dummy molecule to determine length
            global_features.extend(np.zeros(len(global_featurizer.featurize(Chem.MolFromSmiles('C'))[0])))
      else:
        logger.warning(f"{generator} generator is not available in DMPNN")

    # Fix nans in features
    replace_token = 0
    if global_features is not None:
        global_features = np.where(np.isnan(global_features), replace_token, global_features)

    return np.asarray(global_features)

  def _phase_features_generator(self, datapoint: RDKitMol):
    return NotImplementedError

  def _generate_atom_descriptors(self, datapoint: RDKitMol):
    return NotImplementedError
