from phermes import Problem
from phermes import HyperHeuristic
from typing import List
import sys
import numpy as np


class VCP (Problem):
	"""
		Provides the methods to create and solve vertex coloring problems.
	"""
	
	def __init__(self, fileName : str):
		f = open(fileName, "r")
		lines = f.readlines() 
		line = lines[0].split(" ")
		nbNodes = int(line[0].strip())
		self._matrix = np.zeros((nbNodes, nbNodes))
		nbEdges = int(line[1].strip())
		self._nodes = [-1] * nbNodes
		for i in range(0, nbEdges):
			line = lines[i + 1].split(" ")
			x = int(line[0].strip()) - 1
			y = int(line[1].strip()) - 1
			self._matrix[x][y] = 1
			self._matrix[y][x] = 1
		self._maxColor = -1
  
	def getSolution(self) -> List[int]:
		return self._nodes

	def solve(self, heuristic : str) -> None:	
		for i in range(len(self._nodes)):
			nodeId = self._nextNode(heuristic)			
			if nodeId != None:				
				self._nodes[nodeId] = self._nextColor(nodeId)

	def solveHH(self, hyperHeuristic : HyperHeuristic) -> None:	
		for i in range(len(self._nodes)):
			heuristic = hyperHeuristic.getHeuristic(self)
			nodeId = self._nextNode(heuristic)			
			if nodeId != None:				
				self._nodes[nodeId] = self._nextColor(nodeId)
			
	def getObjValue(self) -> float:
		return self._maxColor + 1

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
		else:
			raise Exception("Feature '" + feature + "' is not recognized by the system.")  

# heuristicas
	def _nextNode(self, heuristic : str) -> int:
		selected = None		
		if heuristic == "DEF":
			for i in range(len(self._nodes)):
				if self._nodes[i] == -1:
					selected = i
					break 
		elif heuristic == "DEG":
			value = -sys.maxsize - 1
			for i in range(len(self._nodes)):
				if self._nodes[i] == -1:
					tmp = sum(self._matrix[i])
					if self._nodes[i] == -1 and tmp > value:
						selected = i
						value = tmp
		elif heuristic == "COL_DEG":
			value = -sys.maxsize - 1			
			for i in range(len(self._nodes)):
				if self._nodes[i] == -1:
					deg = 0
					for j in range(len(self._nodes)):
						if self._matrix[i][j] == 1 and self._nodes[j] != -1:
							deg += 1
					if deg > value:
						value = deg
						selected = i
		elif heuristic == "UNCOL_DEG":
			value = -sys.maxsize - 1
			for i in range(len(self._nodes)):
				if self._nodes[i] == -1:
					deg = 0
					for j in range(len(self._nodes)):
						if self._matrix[i][j] == 1 and self._nodes[j] == -1:
							deg += 1
					if deg > value:
						value = deg
						selected = i
		else: 
			raise Exception("Heuristic '" + heuristic + "' is not recognized by the system.")							
		return selected

# siguiente codigo a colorear
	def _nextColor(self, nodeId : str) -> int:		
		forbidden = [None] * len(self._nodes)
		for i in range(len(self._nodes)):
			if self._matrix[nodeId][i] == 1:
				forbidden[i] = self._nodes[i]		
		for i in range(self._maxColor):
			if not (i in forbidden):
				return i
		self._maxColor += 1		
		return self._maxColor

	def __str__(self):
		return str(np.matrix(self._matrix))