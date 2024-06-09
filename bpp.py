from phermes import Problem
from phermes import HyperHeuristic
import sys

# ====================================

class Item:
	"""
		Provides the methods to create and use items for the one-dimensional bin packing problem.
	"""

	def __init__(self, id, length : int):
		self.id = id    
		self.length = length

	def getId(self):
		return self.id

	def getLength(self):
		return self.length  

	def __str__(self):
		return f"({self.id}, {self.length})"

# ====================================

class Bin:
	"""
		Provides the methods to create and use bins for the one-dimensional bin packing problem.
	"""

	def __init__(self, capacity : int):
		"""
			Creates a new instance of Bin
		"""
		self._capacity = capacity    
		self._items = []

	def getCapacity(self) -> int:
		return self._capacity

	def canPack(self, item : Item) -> bool:
		return item.getLength() <= self._capacity

	def pack(self, item : Item) -> None:
		if item.getLength() <= self._capacity:
			self._items.append(item)
			self._capacity -= item.getLength()      
			return True
		return False

	def __str__(self):
		text = "("
		for item in self._items:
			text += str(item)
		text += ")"
		return text

# ====================================

class BPP (Problem):
	"""
		Provides the methods to create and solve one-dimensional bin packing problems.
	"""
	def __init__(self, fileName : str):
		f = open(fileName, "r")
		lines = f.readlines()    
		nbItems = int(lines[0].strip())
		self._capacity = int(lines[1].strip())
		self._items = [None] * nbItems
		for i in range(0, nbItems):
			size = int(lines[i + 2].strip())
			self._items[i] = Item(i, size)    
		self._openBins = []
		self._closedBins = []

	def solve(self, heuristic : str) -> None: 		   
		while self._items:
			item = self._items.pop(0)
			bin = self._selectBin(item, heuristic)
			if bin == None:
				bin = Bin(self._capacity)
				self._openBins.append(bin)
			bin.pack(item)
			if bin.getCapacity() == 0:
				self._openBins.remove(bin)
				self._closedBins.append(bin)
			
	def solveHH(self, hyperHeuristic : HyperHeuristic) -> None:		
		while self._items:
			item = self._items.pop(0)
			heuristic = hyperHeuristic.getHeuristic(self)
			bin = self._selectBin(item, heuristic)
			if bin == None:
				bin = Bin(self._capacity)
				self._openBins.append(bin)
			bin.pack(item)
			if bin.getCapacity() == 0:
				self._openBins.remove(bin)
				self._closedBins.append(bin)

	def getObjValue(self) -> float:
		waste = 0
		for bin in self._openBins:
			waste += ((self._capacity - bin.getCapacity() ) / self._capacity) ** 2
		for bin in self._closedBins:
			waste += ((self._capacity - bin.getCapacity()) / self._capacity) ** 2
		return waste / (len(self._openBins) + len(self._closedBins))		

	def getFeature(self, feature : str) -> float:
		if feature == "OPEN":
			if (len(self._openBins) + len(self._closedBins)) > 0:
				return len(self._openBins) / (len(self._openBins) + len(self._closedBins))
			return 0
		elif feature == "LENGTH":
			values = [0] * len(self._items)
			for i in range(len(self._items)):
				values[i] = self._items[i].getLength()
			if values:
				return (sum(values) / len(values)) / max(values)
			return 0
		elif feature == "SMALL":
			count = 0
			for item in self._items:
				if (item.getLength() < 0.5 * self._capacity):
					count += 1
			if self._items:
				return count / len(self._items)
			return 0
		elif feature == "LARGE":
			count = 0
			for item in self._items:
				if (item.getLength() >= 0.5 * self._capacity):
					count += 1
			if self._items:
				return count / len(self._items)
			return 0
		else:
			raise Exception("Feature '" + feature + "' is not recognized by the system.")

	def _selectBin(self, item : Item, heuristic : str) -> Item:    
		selected = None
		if heuristic == "FFIT":
			for bin in self._openBins:
				if bin.canPack(item):
					selected = bin
					break
			return selected
		elif heuristic == "BFIT":
			waste = sys.maxsize
			for bin in self._openBins:
				if bin.canPack(item):
					tmp = bin.getCapacity() - item.getLength()
					if tmp < waste:
						selected = bin
						waste = tmp					
			return selected
		elif heuristic == "WFIT":
			waste = -sys.maxsize - 1
			for bin in self._openBins:
				if bin.canPack(item):
					tmp = bin.getCapacity() - item.getLength()
					if tmp > waste:
						selected = bin
						waste = tmp
			return selected
		elif heuristic == "AWFIT":
			waste = -sys.maxsize - 1
			waste2 = -sys.maxsize - 1
			selectedTmp = None
			for bin in self._openBins:
				if bin.canPack(item):
					tmp = bin.getCapacity() - item.getLength()
					if tmp > waste:
						selected = selectedTmp
						selectedTmp = bin
						waste2 = waste
						waste = tmp
			if selected == None:
				return selectedTmp
			return selected
		else:
			raise Exception("Heuristic '" + heuristic + "' is not recognized by the system.")

	def __str__(self):
		text = "("
		for item in self._items:
			text += str(item)
		text += ")"
		return text