import torch, threading, misc, ccboard
from helper import args, stack_input, model_predict
import log
from mcts import Node, Tree

logger = log.Logger(args(), init_train_info={'epoch':0, 'loss':[]})
model = logger.create_model(0)
board = ccboard.ChessBoard()
node = Node(board)
tree = Tree(node, model, 0, args().c_puct)
policy = tree.search(1, 8)