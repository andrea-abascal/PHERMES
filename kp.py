from phermes import Problem
from phermes import HyperHeuristic
import sys
import numpy as np

# ====================================

class Item:
  """
    Provides the methods to create and use items for the knapsack problem.
  """

  def __init__(self, id, weight: int, profit : int):
    self._id = id
    self._profit = profit
    self._weight = weight
  
  def getId(self):
    return self._id
  
  def getProfit(self):
    return self._profit

  def getWeight(self):
    return self._weight

  def getProfitPerWeightUnit(self):
    return self._profit / self._weight

  def __str__(self):
    return f"({self._id}, {self._weight}, {self._profit})"

# ====================================

class Knapsack:
  """
    Provides the methods to create and use knapsacks for the knapsack problem.
  """
  
  def __init__(self, capacity : int):
    """
      Creates a new instance of Knapsack
    """
    self._capacity = capacity
    self._profit = 0
    self._items = []    

  def getCapacity(self) -> int:
    return self._capacity
  
  def getProfit(self) -> int:
    return self._profit

  def canPack(self, item : Item) -> bool:
    return item.getWeight() <= self._capacity

  def pack(self, item : Item) -> None:
    if item.getWeight() <= self._capacity:
      self._items.append(item)
      self._capacity = self._capacity - item.getWeight()
      self._profit += item.getProfit()
      return True
    return False
  
  def __str__(self):
    text = "("
    for item in self._items:
      text += str(item)
    text += ")"
    return text

# ====================================

class KP (Problem):
  """
    Provides the methods to create and solve knapsack problems.
  """

  def __init__(self, fileName : str):
    f = open(fileName, "r")
    lines = f.readlines()
    line = lines[0].split(",")
    nbItems = int(line[0].strip())
    self._capacity = int(line[1].strip())
    self._items = [None] * nbItems
    for i in range(0, nbItems):
      line = lines[i + 1].split(",")
      weight = int(line[0].strip())
      profit = int(float(line[1].strip()))
      self._items[i] = Item(i, weight, profit)    
    self._knapsack = Knapsack(self._capacity)
    
  def solve(self, heuristic : str) -> None:
    item = None    
    item = self._nextItem(heuristic)      
    while item != None:    
      self._knapsack.pack(item)
      self._items.remove(item)
      item = self._nextItem(heuristic)
    return self._knapsack
  
  def solveHH(self, hyperHeuristic : HyperHeuristic) -> None:
    item = None
    heuristic = hyperHeuristic.getHeuristic(self)
    item = self._nextItem(heuristic)
    while item != None:      
      self._knapsack.pack(item)
      self._items.remove(item)
      heuristic = hyperHeuristic.getHeuristic(self)
      item = self._nextItem(heuristic)    
  
  def getObjValue(self) -> float:
    return self._knapsack.getProfit()

  def getFeature(self, feature : str) -> float:
    if feature == "WEIGHT":
        values = [0] * len(self._items)
        for i in range(len(self._items)):
          values[i] = self._items[i].getWeight()
        return (sum(values) / len(values)) / max(values)
    elif feature == "PROFIT":
        values = [0] * len(self._items)
        for i in range(len(self._items)):
          values[i] = self._items[i].getProfit()
        return (sum(values) / len(values)) / max(values)
    elif feature == "CORRELATION":
        valuesX = [0] * len(self._items)
        valuesY = [0] * len(self._items)
        for i in range(len(self._items)):
          valuesX[i] = self._items[i].getWeight()
          valuesY[i] = self._items[i].getProfit()
        return np.corrcoef(valuesX, valuesY)[0, 1] / 2 + 0.5
    else:
      raise Exception("Feature '" + feature + "' is not recognized by the system.")
   
  def _nextItem(self, heuristic : str) -> Item:    
    selected = None
    if heuristic == "DEF":
        for item in self._items:          
          if self._knapsack.canPack(item):            
            selected = item             
            break          
        return selected    
    elif heuristic == "MAXP":
        value = -sys.maxsize - 1
        for item in self._items:
          if self._knapsack.canPack(item) and item.getProfit() > value:
            selected = item
            value = item.getProfit()            
        return selected
    elif heuristic == "MAXPW":
        value = -sys.maxsize - 1
        for item in self._items:
          if self._knapsack.canPack(item) and item.getProfitPerWeightUnit() > value:
            selected = item
            value = item.getProfitPerWeightUnit()            
        return selected
    elif heuristic == "MINW":
        value = sys.maxsize
        for item in self._items:
          if self._knapsack.canPack(item) and item.getWeight() < value:
            selected = item
            value = item.getWeight()
        return selected
    elif heuristic == "MARK":
        value = -sys.maxsize - 1
        for item in self._items:
          if self._knapsack.canPack(item) and item.getProfit() * item.getWeight() > value:
            selected = item
            value = item.getProfit() * item.getWeight()
        return selected
    else: 
      raise Exception("Heuristic '" + heuristic + "' is not recognized by the system.")

  def __str__(self):
    text = "Capacity:" + str(self._capacity) + "("
    for item in self._items:
      text += str(item)
    text += ")"
    return text 