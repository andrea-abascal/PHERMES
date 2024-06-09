from __future__ import annotations
from typing import List
from copy import deepcopy
import numpy

# ====================================

class HyperHeuristic:
  """
    Provides a generic definition of the methods to create and use hyper-heuristics.
  """

  def __init__(self, features : List[str], heuristics : List[str]):
    """
        Creates a new instance of HyperHeuristic

        features : List[str] 
        heuristics : List[str]
    """
    self._features = deepcopy(features)
    self._heuristics = deepcopy(heuristics)

  def getFeatures(self) -> List[str]:
    """
      Returns the features used by this hyper-heuristic.
    """
    return deepcopy(self._features)

  def getHeuristics(self) -> List[str]:
    return deepcopy(self._heuristics)

  def getHeuristic(self, problem : Problem) -> str:
    """
      Returns the heuristic recommended for this problem state.
    """
    raise Exception("Method not implemented yet.")

# ====================================

class Problem:
  """
    Provides the basic functionality for all the problems supported by the system.
  """

  def solve(self, heuristic : str) -> None:
    """
      Solves this problem by using a specific heuristic.
    """
    raise Exception("Method not implemented yet.")

  def solveHHA(self, hyperHeuristic : HyperHeuristic) -> None:
    """
      Solves this problem by using a type A hyper-heuristic.
    """
    raise Exception("Method not implemented yet.")

  def solveHHB(self, hyperHeuristic : HyperHeuristic) -> None:
    """
      Solves this problem by using a type B hyper-heuristic.
    """
    raise Exception("Method not implemented yet.")  

  def getFeature(self, feature : str) -> float:
    """
      Returns the current value of a given feature.
    """
    raise Exception("Method not implemented yet.")

  def getObjValue(self) -> float:
    """
      Returns the objective value of the current solution to this problem.
    """
    raise Exception("Method not implemented yet.")
