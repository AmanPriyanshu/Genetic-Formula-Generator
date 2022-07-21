import networkx as nx
from matplotlib import pyplot as plt

class Node:
	def __init__(self, value):
		self.value = value
		self.children = []

	def add_child(self, child_node):
		self.children.append(child_node)

class Converter:
	def __init__(self, operators, operands):
		self.operator_list = [i[0] for i in operators]
		self.operator_counter = [i[1] for i in operators]
		self.operand_list = operands

	def prefix2infix(self, input_list):
		if len(input_list)==1 and input_list[0] in self.operand_list:
			root = Node(input_list[0])
			return root

		operands = []
		while len(input_list)>0:
			symbol = input_list[0]
			input_list = input_list[1:]
			if symbol in self.operand_list:
				operands.append(Node(symbol))
			elif symbol in self.operator_list:
				group = operands[-self.operator_counter[self.operator_list.index(symbol)]:]
				operands = operands[:-self.operator_counter[self.operator_list.index(symbol)]]
				n_op = Node(symbol)
				for n_operand in group:
					n_op.add_child(n_operand)
				operands.append(n_op)
			else:
				print("That symbol doesn't exist!!!")
				return False, None
		if len(operands)!=1:
			return False, None
		else:
			return True, operands[0]

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





if __name__ == '__main__':
	c = Converter([('~', 1), ('*', 2), ('+', 2), ('-', 2), ('/', 2)], ['a', 'b', 'c'])
	valid, root = c.prefix2infix("cc*")
	if valid:
		c.visulaize_tree(root)
	print(valid, root)