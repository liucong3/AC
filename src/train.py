# -*- coding: utf-8 -*-

import torch, threading, misc, ccboard
from helper import args, stack_input, model_predict

epoch = 0
replay_buffer = []
replay_buffer_lock = threading.Lock()
model = None # training model
best_model = None
model_lock = threading.Lock()
best_model_lock = threading.Lock()
gpu_mapping = {}
logger = None

# exploration

exploration_iters = 0

def exploration_thread_func():
	from mcts import exploration
	global best_model, model_lock, gpu_mapping, logger
	thread_name = threading.currentThread().name
	gpu_id = gpu_mapping[thread_name]
	last_best_model = None
	exp_model = None
	has_cuda = torch.cuda.is_available()
	while True:
		if last_best_model != best_model:
			best_model_lock.acquire()
			last_best_model = best_model
			exp_model = logger.clone_model(best_model, gpu_id)
			if has_cuda:
				torch.cuda.synchronize()
			best_model_lock.release()
		history, _, _ = exploration(ccboard.ChessBoard(), [exp_model, exp_model], gpu_id, policy_noise_ratio=args().policy_noise_ratio)
		replay_buffer_lock.acquire()
		for h in history:
			replay_buffer.append(h)
		while len(replay_buffer) > args().replay_buffer_size:
			replay_buffer.pop(0)
		replay_buffer_lock.release()
		global exploration_iters
		exploration_iters += 1
		logger.info('Exploration %d %s %s@GPU%d buffer_size:%d' % (exploration_iters, misc.datetimestr(), thread_name, gpu_id, len(replay_buffer)))

def start_exploration_threads():
	global gpu_mapping
	for gpu_id in args().exploration_gpus:
		for i in range(args().exploration_threads):
			exploration_thread = threading.Thread(target=exploration_thread_func)
			exploration_thread.setDaemon(True)
			gpu_mapping[exploration_thread.name] = gpu_id
			exploration_thread.start()

# train

def anneal_lr(optimizer):
	global logger
	optim_state = optimizer.state_dict()
	optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args().learning_anneal
	optimizer.load_state_dict(optim_state)
	logger.info('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

def train_epoch(has_cuda, gpu_id, optimizer, max_norm=None):
	global logger, model, replay_buffer, replay_buffer_lock
	import numpy
	batch_size = args().batch_size

	replay_buffer_lock.acquire()
	if batch_size > len(replay_buffer): batch_size = len(replay_buffer)
	batch = numpy.random.choice(replay_buffer, batch_size)
	replay_buffer_lock.release()

	repr_list = torch.stack([d.s.nn_board_repr() for d in batch])
	x = stack_input(repr_list)
	target_p = torch.stack([get_policy(d.p, model.policy_size) for d in batch])
	target_v = torch.Tensor([d.v for d in batch])
	return _learn(x, target_p, target_v, optimizer, has_cuda, gpu_id, max_norm)

def get_policy(p, policy_size):
	(x, y), a = p
	policy = torch.zeros(policy_size)
	policy[a, x, y] = 1
	return policy

def _learn(x, target_p, target_v, optimizer, has_cuda, gpu_id, max_norm):
	global logger, model, model_lock
	from torch.autograd import Variable
	mini_batch = 128
	batch_size = x.size(0)
	optimizer.zero_grad()
	total_lost = 0
	if has_cuda:
		target_p = target_p.cuda(gpu_id)
		target_v = target_v.cuda(gpu_id)
	model_lock.acquire()
	for i in range((batch_size - 1) // mini_batch + 1):
		start = i * mini_batch + 1
		end = (i + 1) * mini_batch + 1
		if end > batch_size: end = batch_size
		p, v = model_predict(model, x[start:end], True, has_cuda, gpu_id)
		loss = model.loss(p, v.view(-1), Variable(target_p[start:end]), Variable(target_v[start:end]))
		loss.backward()
		total_lost += loss.data[0]
	if max_norm is not None:
		torch.nn.utils.clip_grad_norm(model.parameters(), max_norm)
	optimizer.step()
	if has_cuda:
		torch.cuda.synchronize()
	model_lock.release()
	return total_lost

def train():
	global epoch, model, logger
	parameters = model.parameters()
	optimizer = torch.optim.SGD(parameters, lr=args().lr, momentum=args().momentum, nesterov=True)
	has_cuda = torch.cuda.is_available()
	gpu_id = args().train_gpu
	for ep in range(args().epochs):
		epoch = ep
		if (epoch + 1) % args().anneal_interval == 0:
			anneal_lr(optimizer)
		if epoch < logger.train_info['epoch']: continue
		loss = train_epoch(has_cuda, gpu_id, optimizer, max_norm=args().max_norm)
		logger.info('Train epoch: %d, Time: %s, Loss: %.5f' % (epoch, misc.datetimestr(), loss))
		logger.train_info['loss'].append(loss)
		if args().plot:
			logger.plot_progress()

# evaluation

def compare_models(iters, eval_model, gpu_id):
	from mcts import exploration
	global logger, best_model
	wins = 0
	has_cuda = torch.cuda.is_available()
	best_model_lock.acquire()
	old_model = logger.clone_model(best_model, gpu_id)
	if has_cuda:
		torch.cuda.synchronize()
	best_model_lock.release()
	iters2 = 0
	for i in range(args().evaluation_games):
		_, winner, _ = exploration(ccboard.ChessBoard(), [old_model, eval_model], gpu_id, policy_noise_ratio=args().policy_noise_ratio)
		if winner is not None:
			wins += winner
		iters2 += 1
		logger.info("Evaluation %d:%d %s New model %s" % (iters, iters2, misc.datetimestr(), 'wins' if winner == 1 else 'lose'))
	return wins / float(args().evaluation_games)

def evaluation(epoch, iters, gpu_id):
	global logger, model, best_model, model_lock
	has_cuda = torch.cuda.is_available()
	model_lock.acquire()
	eval_model = logger.clone_model(model, gpu_id)
	if has_cuda:
		torch.cuda.synchronize()
	model_lock.release()
	wins = compare_models(iters, eval_model, gpu_id)
	logger.info(">>> Evaluation %d %s New model wins %.0f%%" % (iters, misc.datetimestr(), wins * 100))
	if wins < 0.55: return
	best_model_lock.acquire()
	best_model = eval_model
	logger.train_info['epoch'] = epoch
	logger.save_model(best_model)
	best_model_lock.release()
	logger.save_train_info()

def evaluation_thread_func():
	global epoch
	gpu_id = args().train_gpu
	iters = 0
	while True:
		iters += 1
		evaluation(epoch, iters, gpu_id)

def start_evaluation_threads():
	evaluation_thread = threading.Thread(target=evaluation_thread_func)
	evaluation_thread.setDaemon(True)
	evaluation_thread.start()	

# main

def main():
	global logger, model, best_model, replay_buffer
	import log
	logger = log.Logger(args(), init_train_info={'epoch':0, 'loss':[]})
	logger.info('\n'.join([arg_name + ': ' + str(arg_val) for arg_name, arg_val in args().__dict__.items()]))
	model = logger.create_model(args().train_gpu)
	best_model = logger.clone_model(model, args().train_gpu)

	start_exploration_threads()
	from time import sleep
	while True:
		replay_buffer_lock.acquire()
		length = len(replay_buffer)
		replay_buffer_lock.release()
		if length >= args().batch_size: break
		sleep(1)
	start_evaluation_threads()
	train()

if __name__ == '__main__':
	main()
