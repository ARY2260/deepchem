import torch
import torch.nn as nn
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.feat import GraphData
from typing import Any


class DMPNN(nn.Module):
  """
  # TODO: implement the DMPNN class, which is an internal TorchModel class for DMPNNModel.
  """
  def __init__(self):
    return NotImplementedError

class _MapperDMPNN:
  """
  This class is a helper class for DMPNNModel class to generate concatenated feature vector and mapping.

  `self.f_ini_atoms_bonds_zero_padded` is the concatenated feature vector which contains
  concatenation of initial atom and bond features.

  `self.mapping` is the mapping which maps bond index to 'array of indices of the bonds'
  incoming at the initial atom of the bond (excluding the reverse bonds)
  """

  def __init__(self, graph: GraphData):
    """
    Parameters
    ----------
    graph: GraphData
      GraphData object.
    """
    self.num_atoms = graph.num_nodes
    self.num_atom_features = graph.num_node_features
    self.num_bonds = graph.num_edges
    self.num_bond_features = graph.num_edge_features
    self.atom_features = graph.node_features
    self.bond_features = graph.edge_features
    self.bond_index = graph.edge_index

    # mapping from bond index to the index of the atom (where the bond is coming from)
    self.bond_to_ini_atom: np.ndarray

    # mapping from bond index to concat(in_atom, bond) features
    self.f_ini_atoms_bonds: np.ndarray = None

    # mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond (excluding the reverse bonds)
    self.mapping: np.ndarray = None

    if self.num_bonds == 0:
      self.f_ini_atoms_bonds = np.zeros((1, self.num_atom_features + self.num_bond_features))
      self.mapping = np.asarray([[-1]], dtype=int)

    else:
      self.bond_to_ini_atom = self.bond_index[0]
      self._get_f_ini_atoms_bonds() # its zero padded at the end
      self._generate_mapping()

  def _get_f_ini_atoms_bonds(self):
    """
    Method to get `self.f_ini_atoms_bonds`
    """
    self.f_ini_atoms_bonds = np.hstack((self.atom_features[self.bond_to_ini_atom], self.bond_features))

    # zero padded at the end
    self.f_ini_atoms_bonds = np.pad(self.f_ini_atoms_bonds, (0,1))

  def _generate_mapping(self):
    """
    Generate mapping which maps bond index to 'array of indices of the bonds'
    incoming at the initial atom of the bond (reverse bonds are not considered).

    Steps:
    - Generate `atom_to_incoming_bonds` matix.
    - Get mapping based on `atom_to_incoming_bonds` and `self.bond_to_ini_atom`.
    - Replace reverse bond indices with -1.
    """
    # mapping from atom index to list of indicies of incoming bonds
    atom_to_incoming_bonds = self._get_atom_to_incoming_bonds()

    # get mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond
    self.mapping = atom_to_incoming_bonds[self.bond_to_ini_atom]
    self._replace_rev_bonds()

  def _get_atom_to_incoming_bonds(self):
    """
    Method to get atom_to_incoming_bonds mapping
    """
    # mapping from bond index to the index of the atom (where the bond if going to)
    bond_to_final_atom = self.bond_index[1]
    
    a2b = []
    for i in range(self.num_atoms):
      a2b.append(list(np.where(bond_to_final_atom==i)[0]))
    
    max_num_bonds: int = max(
        1,
        max(
            len(incoming_bonds)
            for incoming_bonds in self.atom_to_incoming_bonds))
    
    a2b = [self.atom_to_incoming_bonds[a] + [-1] *
        (max_num_bonds - len(self.atom_to_incoming_bonds[a]))
        for a in range(self.num_atoms + 1)]
    
    return np.asarray(a2b, dtype=int)

  def _replace_rev_bonds(self):
    """
    Method to get b2revb and replace the reverse bond indicies with -1 in mapping.
    """
    # mapping from bond index to the index of the reverse bond
    b2revb = np.empty(self.num_bonds, dtype=int)
    for i in range(self.num_bonds):
      if i%2==0:
        b2revb[i] = i+1
      else:
        b2revb[i] = i-1
    
    for count, i in enumerate(b2revb):
      self.mapping[count][np.where(self.mapping[count] == i)] = -1


class DMPNNModel(TorchModel):
  """
  # TODO: implement the main DMPNN model class utilizing DMPNN class.
  """
  def __init__(self):
    return NotImplementedError
