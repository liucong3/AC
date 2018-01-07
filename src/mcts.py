import torch
from helper import args

class Edge(object):

	def __init__(self, prev_node, a, p):
		# see page 8 of the nature paper
		# $prev_node.s$ and $self.next$ are of type Board
		# With $a$, we can call prev_node.s.next_board(*a)
		self.prev_node = prev_node
		self.a = a
		self.n = 0
		self.w = 0
		self.p = p
		self.next_node = None

	# def is_leaf(self):
	# 	return self.next_node is None

	def q(self):
		if self.n == 0:
			return 0
		return self.w / self.n

	def u(self, c_puct):
		sum_n = 0
		for e in self.prev_node.next_edges.values():
			sum_n += e.n
		if sum_n == 0: sum_n == 1
		import math
		return c_puct * self.p * math.sqrt(sum_n) / (1 + self.n)


class Node(object):

	def __init__(self, s, prev_edge=None):
		self.s = s
		self.prev_edge = prev_edge
		self.next_edges = None
		self.n = 0

	def expand(self, p, v):
		self.next_edges = {}
		self.v = v[0]
		if self.s.is_terminated():
			if self.s.is_winner(): raise
			self.v = -1
		else:
			actions, p = self.s.nn_actions(p) # [ (pos, action), ... ]
			for i in range(len(actions)):
				action = actions[i]
				self.next_edges[action] = Edge(self, action, p[i])
		self._propagate_w_backward(self.v)

	def _propagate_w_backward(self, v):
		v = -v #
		if self.prev_edge is not None:
			self.prev_edge.w += v
			self.prev_edge.n += 1
			self.prev_edge.prev_node.n += 1
			self.prev_edge.prev_node._propagate_w_backward(v)


class Tree(object):

	def __init__(self, root_node, model, gpu_id, c_puct):
		self.model = model
		self.gpu_id = gpu_id
		self.root_node = root_node
		self.c_puct = c_puct

	def search(self, batch_size=8, simulations=128):
		batch = []
		if self.root_node.next_edges is None:
			batch.append(self.root_node)
		while self.root_node.n < simulations:
			to_expand = self._search_node(self.root_node)
			if to_expand is not None:
				batch.append(to_expand)
			if len(batch) == batch_size or to_expand is None:
				if len(batch) == 0: break
				self._expand_nodes(batch)
				batch = []
		return self._get_policy()

	def _get_policy(self):
		policy = torch.zeros(*self.model.policy_size)
		for action, edge in self.root_node.next_edges.items():
			(x, y), a = action
			# print '>', action, edge.n, edge.next_node is not None
			policy[a, x, y] = edge.n
		return policy

	def _expand_nodes(self, batch):
		batch_size = len(batch)
		repr_list = torch.stack([node.s.nn_board_repr() for node in batch])
		is_cuda = torch.cuda.is_available()
		from helper import stack_input, model_predict
		x = stack_input(repr_list)
		p, v = model_predict(self.model, x, False, is_cuda, self.gpu_id)
		p = [ x.squeeze(0) for x in p.chunk(batch_size) ]
		v = [ x.squeeze(0) for x in v.chunk(batch_size) ]

		for i in range(batch_size):
			node = batch[i]
			node.expand(p[i], v[i])

	# Returns a node to expand, None if no node to expand
	def _search_node(self, node):
		if node.next_edges is None: # the node has not been expanded
			return None
		if node.s.is_terminated(): # terminated node should not be searched
			node._propagate_w_backward(node.v)
			return None
		edges = list(node.next_edges.values())
		edges.sort(key=lambda e: e.q() + e.u(self.c_puct), reverse=True)
		for edge in edges:
			if edge.next_node is None:
				s = node.s.get_next_board(*edge.a)[0]
				s.switch_players()
				edge.next_node = Node(s, edge)
				return edge.next_node
			to_expand = self._search_node(edge.next_node)
			if to_expand is not None:
				return to_expand
		return None




def default_tau_func(step):
	return 1 if step < 50 else 0

# returns a normalized policy
def _mcts_policy(node, model, gpu_id, tau):
	tree = Tree(node, model, gpu_id, args().c_puct)
	policy = tree.search(args().search_batch_size, args().search_simulations)
	if tau == 0:
		policy = (policy == policy.max()).type_as(policy)
	else:
		policy = policy.pow(1 / tau)
	policy = policy / policy.sum()
	return policy

def _next_action(board, policy, noice_ratio):
	if noice_ratio > 0:
		noice = torch.rand(policy.size())
		noice = noice * (noice_ratio / noice.sum())
		policy = policy * (1 - noice_ratio) + noice
	actions, policy = board.nn_actions(policy)
	import numpy
	policy = numpy.array(policy)
	sum_ = policy.sum()
	if sum_ == 0:
		policy[:] = 1.0 / len(policy)
	else:
		policy = policy / float(sum_)
	policy = policy.tolist()
	# print policy
	index = numpy.random.choice(range(len(policy)), size=1, p=policy)[0]
	# print '>>>>', policy[index]
	return actions[index]

class Experience(object):

	def __init__(self, s, p, v):
		self.s = s
		self.p = p
		self.v = v

def next_action_for_evaluation(model, board):
	policy = _mcts_policy(Node(board), model, 0, 0)
	return _next_action(board, policy, 0)

def exploration(board, models, gpu_id, tau_func=default_tau_func, policy_noise_ratio=0, resign=None, logger=None):
	import misc, json, sys
	history = []
	cur_node = Node(board)
	winner = None
	# from tqdm import trange
	game_name = misc.datetimestr()
	for step in range(args().max_game_steps):
		# misc.progress_bar(step, args().max_game_steps, game_name)

		policy = _mcts_policy(cur_node, models[step % 2], gpu_id, tau_func(step) )
		pos, action = _next_action(cur_node.s, policy, policy_noise_ratio)
		#  history.append( Experience(cur_node.s, policy, cur_node.v) ) # $v$ is here for resignation check later 
		history.append( Experience(cur_node.s, (pos, action), cur_node.v) )
		if resign is not None and v < resign:
			winner = (step + 1) % 2 # current player losses 
			# sys.stderr.write('\n')
			break

		if logger is not None:
			logger.log_game_action(game_name, pos, board.action_list()[action])

		cur_node = cur_node.next_edges[(pos, action)].next_node
		cur_node.prev_edge = None
		if cur_node.s.is_terminated():
			# history.append( Experience(cur_node.s, null, cur_node.v) )
			# the current step loss if the next step wins
			next_step_wins = cur_node.s.is_winner()
			winner = (step + 1 if next_step_wins else step) % 2
			# sys.stderr.write('\n')
			break

	if logger is not None:
		logger.end_log_game_action(game_name)
		
	# fill scores
	min_winner_score = 1
	for step in range(len(history)):
		if winner is None:
			history[step].v = 0
		elif step % 2 == winner:
			if min_winner_score > history[step].v: 
				min_winner_score = history[step].v
			history[step].v = 1
		else:
			history[step].v = -1
	return history, winner, min_winner_score

	