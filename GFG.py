import numpy as np
import inspect
from common_operators import add, sub, mul, div, neg
from copy import deepcopy
import time
from tqdm import trange
import networkx as nx
from matplotlib import pyplot as plt

class Node:
	def __init__(self, value):
		self.value = value
		self.children = []

	def add_child(self, child_node):
		self.children.append(child_node)

class GeneticFormulaGenerator:
	def __init__(self, operators, operands, target, max_genome_sequence=10, min_genome_sequence=None, max_iters_sequence_resolver=100):
		self.operators = operators
		self.target = target
		self.operands = operands
		if min_genome_sequence is None:
			min_genome_sequence = 1
		self.min_genome_sequence = min_genome_sequence
		self.max_genome_sequence = max_genome_sequence
		self.operator_counts = [len(inspect.getfullargspec(i).args) for i in self.operators]
		self.max_iters_sequence_resolver = max_iters_sequence_resolver

	def generate_single_random_sequence(self):
		num_operators = np.random.randint(low=0, high=self.max_genome_sequence//2)
		if num_operators==0:
			return int(np.random.randint(low=0, high=len(self.operands)))
		at_hand_operands = []
		for i in range(num_operators):
			operator_idx = int(np.random.randint(low=0, high=len(self.operators)))
			all_operands_considered = list(range(len(self.operands)))+at_hand_operands
			operand_indices = np.random.choice(np.arange(0, len(all_operands_considered)), replace=False, size=self.operator_counts[operator_idx])
			new_tree = (operator_idx, [i if i<len(self.operands) else at_hand_operands[i-len(self.operands)] for i in operand_indices])
			tmp_at_hand_operands = []
			for index, tmp_operand in enumerate(at_hand_operands):
				if index+len(self.operands) in operand_indices:
					continue
				else:
					tmp_at_hand_operands.append(tmp_operand)
			at_hand_operands = tmp_at_hand_operands
			at_hand_operands.append(new_tree)
		count_iter = 0
		while len(at_hand_operands)>1:
			count_iter += 1
			operator_idx = int(np.random.randint(low=0, high=len(self.operators)))
			try:
				operand_indices = np.random.choice(np.arange(0, len(at_hand_operands)), replace=False, size=self.operator_counts[operator_idx])
			except:
				continue
			new_tree = (operator_idx, [at_hand_operands[i] for i in operand_indices])
			tmp_at_hand_operands = []
			for index, tmp_operand in enumerate(at_hand_operands):
				if index in operand_indices:
					continue
				else:
					tmp_at_hand_operands.append(tmp_operand)
			at_hand_operands = tmp_at_hand_operands
			at_hand_operands.append(new_tree)
			if count_iter>=self.max_iters_sequence_resolver:
				count_iter = 0
				at_hand_operands.pop(0)
		return at_hand_operands[0]

	def get_all_operands(self, tree):
		if type(tree) != tuple:
			operands = [tree]
		else:
			operands = deepcopy(tree[1])
			for operand in tree[1]:
				operands += self.get_all_operands(operand)
		return operands

	def make_offspring(self, parent_a, parent_b):
		all_possible_operands = self.get_all_operands(parent_a) + self.get_all_operands(parent_b)
		all_possible_operands = [operand for operand in all_possible_operands if type(operand)==tuple] + list(set([operand for operand in all_possible_operands if type(operand)!=tuple]))
		at_hand_operands = []
		while len(at_hand_operands)==0:
			at_hand_operands = [all_possible_operands[i] for i in np.random.choice(np.arange(0, len(all_possible_operands)), replace=False, size=1+min(int(np.random.randint(low=1, high=self.max_genome_sequence//2)), len(all_possible_operands)//2))]
		count_iter = 0
		while len(at_hand_operands)>1:
			count_iter += 1
			operator_idx = int(np.random.randint(low=0, high=len(self.operators)))
			try:
				operand_indices = np.random.choice(np.arange(0, len(at_hand_operands)), replace=False, size=self.operator_counts[operator_idx])
			except:
				continue
			new_tree = (operator_idx, [at_hand_operands[i] for i in operand_indices])
			tmp_at_hand_operands = []
			for index, tmp_operand in enumerate(at_hand_operands):
				if index in operand_indices:
					continue
				else:
					tmp_at_hand_operands.append(tmp_operand)
			at_hand_operands = tmp_at_hand_operands
			at_hand_operands.append(new_tree)
			if count_iter>=self.max_iters_sequence_resolver:
				count_iter = 0
				at_hand_operands.pop(0)
		return at_hand_operands[0]

	def compute_result(self, sequence):
		if type(sequence)==tuple:
			operator_idx = sequence[0]
			operands = []
			for val in sequence[1]:
				operands.append(self.compute_result(val))
			out = self.operators[operator_idx](*operands)
		else:
			out = self.operands[int(sequence)]
		return out

	def compute_loss(self, sequence):
		out = self.compute_result(sequence)
		return np.sqrt(np.mean(np.square(out - self.target)))

	def single_exploration(self, pop):
		losses = [self.compute_loss(individual) for individual in pop]
		indices = np.argsort(losses)
		parent_pop = [pop[i] for i in indices[:int(0.4*len(indices))]]
		child_pop = [self.make_offspring(parent_pop[i], parent_pop[i+1]) for i in range(0, len(parent_pop), 2)]
		pop = parent_pop+child_pop+[self.generate_single_random_sequence() for _ in range(len(pop) - len(parent_pop) - len(child_pop))]
		return pop

	def search(self, pop_size=1000, gen_iters=10):
		pop = [self.generate_single_random_sequence() for _ in range(pop_size)]
		bar = trange(gen_iters)
		for epoch in bar:
			pop = self.single_exploration(pop)
			losses = [self.compute_loss(individual) for individual in pop]
			bar.set_description(str({"epoch": epoch+1, "mean_loss": round(np.mean(losses[:int(0.4*len(losses))]), 3), "min_loss": round(np.min(losses[:int(0.4*len(losses))]), 3)}))
			time.sleep(1)
		bar.close()
		sorted_pop = [pop[i] for i in np.argsort(losses)]
		return sorted_pop

	def get_representation(self, node, G, parent_op=None, repititions=[], count_repititions=[], prev_depth=0):
		if node.value in repititions:
			val = node.value+" ("+str(count_repititions[repititions.index(node.value)])+")"
			count_repititions[repititions.index(node.value)] += 1
		else:
			val = node.value
			repititions.append(val)
			count_repititions.append(1)
		G.add_node(val)
		self.pos.update({val:[len(list(self.pos)),-(prev_depth+1)]})
		if parent_op is not None:
			G.add_edge(val, parent_op)
		for child in node.children:
			self.get_representation(child, G, val, repititions, count_repititions, prev_depth+1)

	def visulaize_tree(self, root):
		self.pos = {}
		G = nx.DiGraph()
		self.get_representation(root, G)
		pos = {}
		for key, item in self.pos.items():
			pos[key] = [item[0], item[1]]
		print(pos)
		nx.draw(G, pos=pos, with_labels=True, node_size=1500)
		plt.show()

	def make_tree_representation(self, sequence, operand_reps):
		if type(sequence)!=tuple:
			root = Node(operand_reps[sequence])
		else:
			root = Node(self.operators[sequence[0]].__name__)
			for val in sequence[1]:
				root.add_child(self.make_tree_representation(val, operand_reps=operand_reps))
		return root

if __name__ == '__main__':
	N = 5000
	a = np.random.uniform(low=-1, high=1, size=N)
	b = np.random.normal(loc=0.0, scale=5, size=N)
	c = np.random.randint(low=0, high=5, size=N)
	d = np.ones(shape=(N,))*5
	y = (-b)*(c/d)

	gfg = GeneticFormulaGenerator(operators=[add, sub, mul, div, neg], operands=[a, b, c, d], target=y, max_genome_sequence=10)
	sorted_pop = gfg.search()
	best_pop = sorted_pop[0]
	root = gfg.make_tree_representation(best_pop, operand_reps=['a', 'b', 'c', 'd'])
	gfg.visulaize_tree(root)