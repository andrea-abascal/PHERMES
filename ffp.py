from phermes import Problem
from phermes import HyperHeuristic
from typing import List
import sys
import numpy as np

class FFP (Problem):

	"""
		Provides the methods to create and solve one-dimensional bin packing problems.
	"""

	def __init__(self, fileName : str):
		f = open(fileName, "r")
		lines = f.readlines() 
		line = lines[0].split(" ")
		nbNodes = int(line[0].strip())
		self._matrix = np.zeros((nbNodes, nbNodes))
		nbEdges = int(line[1].strip())
		self._nodes = [0] * nbNodes
		for i in range(0, nbEdges):
			line = lines[i + 1].split(" ")
			x = int(line[0].strip()) - 1
			y = int(line[1].strip()) - 1
			self._matrix[x][y] = 1
			self._matrix[y][x] = 1
		self._nodes[0] = -1		
		self._costs = np.random.random(nbNodes)		

	def getSolution(self) -> List[int]:
		return self._nodes

	def solve(self, heuristic : str ) -> None:
		for i in range(len(self._nodes)):
			for j in range(1):
				nodeId = self._nextNode(heuristic)			
				if nodeId != None:			
					self._nodes[nodeId] = 1							
			nodes = np.copy(self._nodes)	
			for j in range(len(nodes)):
				if (nodes[j] == -1):
					for k in range(len(self._matrix[j])):
						if self._matrix[j][k] == 1 and nodes[k] == 0:							
							self._nodes[k] = -1											
	
	def solveHH(self, hyperHeuristic : HyperHeuristic) -> None:
		for i in range(len(self._nodes)):
			for j in range(1):
				heuristic = hyperHeuristic.getHeuristic(self)
				nodeId = self._nextNode(heuristic)			
				if nodeId != None:			
					self._nodes[nodeId] = 1							
			nodes = np.copy(self._nodes)	
			for j in range(len(nodes)):
				if (nodes[j] == -1):
					for k in range(len(self._matrix[j])):
						if self._matrix[j][k] == 1 and nodes[k] == 0:							
							self._nodes[k] = -1
			
	def getObjValue(self) -> float:
		cost = 0
		for i in range(len(self._nodes)):
			if self._nodes[i] == -1:
				cost += self._costs[i]
		return cost

	def getFeature(self, feature : str) -> float:
		if feature == "DENSITY":
			value = 0
			for i in range(len(self._nodes)):
				value += sum(self._matrix[i])
			n = len(self._nodes)
			return value / (n * (n -1))
		elif feature == "MAX_DEG":
			maxDeg = -sys.maxsize - 1			
			for i in range(len(self._nodes)):
				value = sum(self._matrix[i])
				if value > maxDeg:
					maxDeg = value				
			return maxDeg / len(self._nodes)
		elif feature == "MIN_DEG":
			minDeg = sys.maxsize			
			for i in range(len(self._nodes)):
				value = sum(self._matrix[i])
				if value < minDeg:
					minDeg = value				
			return minDeg / len(self._nodes)
		elif feature == "COST":
			return sum(self._costs) / len(self._nodes)
		else:
			raise Exception("Feature '" + feature + "' is not recognized by the system.")  

	def _nextNode(self, heuristic : str) -> int:
		selected = None		
		if heuristic == "DEF":
			for i in range(len(self._nodes)):
				if self._nodes[i] == 0:
					selected = i
					break 
		elif heuristic == "DEG":
			value = -sys.maxsize - 1
			for i in range(len(self._nodes)):
				if self._nodes[i] == 0:
					tmp = sum(self._matrix[i])
					if tmp > value:
						selected = i
						value = tmp
		elif heuristic == "RISK_DEG":
			value = -sys.maxsize - 1			
			for i in range(len(self._nodes)):
				if self._nodes[i] == 0:
					deg = 0
					for j in range(len(self._nodes)):
						if self._matrix[i][j] == 1 and self._nodes[j] != -1:
							deg += 1
					if deg > value:
						value = deg
						selected = i
		else: 
			raise Exception("Heuristic '" + heuristic + "' is not recognized by the system.")							
		return selected