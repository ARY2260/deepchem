"""
Basic molecular features.
"""

import numpy as np
import scipy.stats as st
import logging

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.rdkit_utils import DescriptorsNormalizationParameters as DNP

logger = logging.getLogger(__name__)


class RDKitDescriptors(MolecularFeaturizer):
  """RDKit descriptors.

  This class computes a list of chemical descriptors like
  molecular weight, number of valence electrons, maximum and
  minimum partial charge, etc using RDKit.

  Attributes
  ----------
  descriptors: List[str]
    List of RDKit descriptor names used in this class.

  Note
  ----
  This class requires RDKit to be installed.

  Examples
  --------
  >>> import deepchem as dc
  >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
  >>> featurizer = dc.feat.RDKitDescriptors()
  >>> features = featurizer.featurize(smiles)
  >>> type(features[0])
  <class 'numpy.ndarray'>
  >>> features[0].shape
  (208,)

  """

  def __init__(self, use_fragment=True, ipc_avg=True, is_normalized=False, use_bcut2d=True):
    """Initialize this featurizer.

    Parameters
    ----------
    use_fragment: bool, optional (default True)
      If True, the return value includes the fragment binary descriptors like 'fr_XXX'.
    ipc_avg: bool, optional (default True)
      If True, the IPC descriptor calculates with avg=True option.
      Please see this issue: https://github.com/rdkit/rdkit/issues/1527.
    is_normalized: bool, optional (default False)
      If True, the return value contains normalized features.
    use_bcut2d: bool, optional (default True)
      
    """
    self.use_fragment = use_fragment
    self.is_normalized = is_normalized
    self.ipc_avg = ipc_avg
    self.descriptors = []
    self.descList = []
    self.normalized_desc = {}
    self.use_bcut2d = use_bcut2d

    # initialize
    if len(self.descList) == 0:
      try:
        from rdkit.Chem import Descriptors
        for descriptor, function in Descriptors.descList:
          if self.use_fragment is False and descriptor.startswith('fr_'):
            continue
          if self.use_bcut2d is False and descriptor.startswith('BCUT2D_'):
            continue
          self.descriptors.append(descriptor)
          self.descList.append((descriptor, function))
      except ModuleNotFoundError:
        raise ImportError("This class requires RDKit to be installed.")

    # check initialization
    assert len(self.descriptors) == len(self.descList)
    
    if is_normalized:
      self._make_normalised_func_dict()

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
    """
    Calculate RDKit descriptors.

    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      1D array of RDKit descriptors for `mol`.
      The length is `len(self.descriptors)`.
    """
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )

    features = []
    if not self.is_normalized:
      for desc_name, function in self.descList:
        if desc_name == 'Ipc' and self.ipc_avg:
          feature = function(datapoint, avg=True)
        else:
          feature = function(datapoint)
        features.append(feature)
    else:
      for desc_name, function in self.descList:
        if desc_name == 'Ipc' and self.ipc_avg:
          feature = function(datapoint, avg=True)
        else:
          feature = function(datapoint)
        # if (desc_name in self.normalized_desc):
        #   feature = self.normalized_desc[desc_name](feature)
        # else:
        #   logger.warning("No normalization for %s", desc_name)
        try:
          feature = self.normalized_desc[desc_name](feature)
        except KeyError:
          logger.warning("No normalization for %s. Feature removed!", desc_name)
          self.descriptors.remove(desc_name)
          continue
        features.append(feature)
    return np.asarray(features)
  
  def _make_normalised_func_dict(self):
    """
      Copyright (c) 2018-2021, Novartis Institutes for BioMedical Research Inc.
      All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials provided
          with the distribution.
        * Neither the name of Novartis Institutes for BioMedical Research Inc.
          nor the names of its contributors may be used to endorse or promote
          products derived from this software without specific prior written
          permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    for name, (distributions_, params, minV, maxV, avg, std) in DNP.desc_norm_params.items():
      arg = params[:-2]
      loc = params[-2]
      scale = params[-1]

      distributions_ = getattr(st, distributions_)

      # cdf => cumulative density functions
      # make the cdf with the parameters
      def cdf(v, distributions_=distributions_, arg=arg,loc=loc,scale=scale,minV=minV,maxV=maxV):
          v = distributions_.cdf(np.clip(v, minV, maxV), loc=loc, scale=scale, *arg)
          return np.clip(v, 0., 1.)

      self.normalized_desc[name] = cdf