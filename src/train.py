# -*- coding: utf-8 -*-

import torch, misc, ccboard
from helper import args, model_predict_train

# exploration

def try_reload_model(model_path, best_model, gpu_id, proc_id, iters):
	import os
	from log import Logger
	cur_model_path = Logger.lastest_model_path(args().logdir)
	if best_model is None or model_path != cur_model_path:
		model_path = cur_model_path
		best_model = Logger._load_model(model_path, gpu_id)
		print('Exploration GPU%d-%d it:%d %s model updated' % (gpu_id, proc_id, iters, misc.datetimestr()))
	return model_path, best_model 

def exploration_process_func(gpu_id, proc_id, queue):
	import numpy
	numpy.random.seed(gpu_id * 101 + proc_id)
	from mcts import exploration
	model_path = None
	best_model = None
	has_cuda = torch.cuda.is_available()
	iters = 0
	while True:
		iters += 1
		model_path, best_model = try_reload_model(model_path, best_model, gpu_id, proc_id, iters)
		history, _, _ = exploration(ccboard.ChessBoard(), [best_model, best_model], gpu_id, policy_noise_ratio=args().policy_noise_ratio)
		print('Exploration GPU%d-%d it:%d %s history_size:%d' % (gpu_id, proc_id, iters, misc.datetimestr(), len(history)))
		queue.put(history)

def start_exploration_processes(ctx, queue):
	for gpu_id in args().exploration_gpus:
		for proc_id in range(args().exploration_processes):
			process = ctx.Process(target=exploration_process_func, args=(gpu_id, 1 + proc_id, queue))
			process.daemon = True
			process.start()

# train

epoch = 0

def anneal_lr(optimizer, logger):
	optim_state = optimizer.state_dict()
	optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args().learning_anneal
	optimizer.load_state_dict(optim_state)
	logger.info('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

def train_epoch(replay_buffer, has_cuda, gpu_id, optimizer, model, model_lock, max_norm):
	import numpy
	batch_size = args().batch_size
	if batch_size > len(replay_buffer): batch_size = len(replay_buffer)
	batch = numpy.random.choice(replay_buffer, batch_size)
	x = torch.stack([d.s.nn_board_repr() for d in batch])
	target_p = torch.Tensor([d.p for d in batch]).long()
	target_v = torch.Tensor([d.v for d in batch]).unsqueeze(-1)
	return _learn(x, target_p, target_v, has_cuda, gpu_id, optimizer, model, model_lock, max_norm)

def _learn(x, target_p, target_v, has_cuda, gpu_id, optimizer, model, model_lock, max_norm):
	from torch.autograd import Variable
	mini_batch = args().train_mini_batch
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
		p, v = model_predict_train(model, x[start:end], has_cuda, gpu_id)
		loss = model.loss(p, v, Variable(target_p[start:end]), Variable(target_v[start:end]))
		loss.backward()
		total_lost += loss.data[0]
	if max_norm is not None:
		torch.nn.utils.clip_grad_norm(model.parameters(), max_norm)
	optimizer.step()
	if has_cuda:
		torch.cuda.set_device(gpu_id)
		torch.cuda.synchronize()
	model_lock.release()
	return total_lost

def train(replay_buffer, queue, model, model_lock, logger):
	global epoch
	parameters = model.parameters()
	optimizer = torch.optim.SGD(parameters, lr=args().lr, momentum=args().momentum, nesterov=True)
	has_cuda = torch.cuda.is_available()
	gpu_id = args().train_gpu
	max_norm = args().max_norm
	for ep in range(args().epochs):
		epoch = ep
		if (epoch + 1) % args().anneal_interval == 0:
			anneal_lr(optimizer, logger)
		if epoch < logger.train_info['epoch']: continue
		while not queue.empty():
			replay_buffer.extend(queue.get())
		while len(replay_buffer) > args().replay_buffer_size: replay_buffer.pop()
		loss = train_epoch(replay_buffer, has_cuda, gpu_id, optimizer, model, model_lock, max_norm)
		logger.info('Train epoch: %d, Buffer:%d, Time: %s, Loss: %.5f' % (epoch, len(replay_buffer), misc.datetimestr(), loss))
		logger.train_info['loss'].append(loss)
		if args().plot:
			logger.plot_progress()

# evaluation

def compare_models(iters, eval_model, best_model, gpu_id, logger):
	from mcts import exploration
	wins = 0
	has_cuda = torch.cuda.is_available()
	iters2 = 0
	for i in range(args().evaluation_games):
		_, winner, _ = exploration(ccboard.ChessBoard(), [best_model, eval_model], gpu_id, policy_noise_ratio=args().policy_noise_ratio)
		if winner is not None:
			wins += winner
		iters2 += 1
		logger.info("Evaluation %d:%d %s New model %s" % (iters, iters2, misc.datetimestr(), 'wins' if winner == 1 else 'lose'))
	return wins / float(args().evaluation_games)

def evaluation(iters, best_model, model, model_lock, logger):
	has_cuda = torch.cuda.is_available()
	gpu_id = args().train_gpu
	model_lock.acquire()
	eval_model = logger.clone_model(model, gpu_id)
	model_lock.release()
	wins = compare_models(iters, eval_model, best_model, gpu_id, logger)
	logger.info(">>> Evaluation %d %s New model wins %.0f%%" % (iters, misc.datetimestr(), wins * 100))
	if wins < 0.55: return best_model
	global epoch
	best_model = eval_model
	logger.train_info['epoch'] = epoch
	logger.save_train_info()
	logger.save_model(best_model)
	return best_model

def evaluation_thread_func(model, model_lock, logger):
	iters = 0
	best_model = model
	while True:
		iters += 1
		best_model = evaluation(iters, best_model, model, model_lock, logger)

def start_evaluation_threads(*args):
	import threading
	evaluation_thread = threading.Thread(target=evaluation_thread_func, args=args)
	evaluation_thread.setDaemon(True)
	evaluation_thread.start()	

# main

def main():
	import log
	logger = log.Logger(args(), init_train_info={'epoch':0, 'loss':[]})
	logger.info('\n'.join([arg_name + ': ' + str(arg_val) for arg_name, arg_val in args().__dict__.items()]))
	model = logger.create_model(args().train_gpu)
	if logger.get_model_path() is None:
		logger.save_model(model)

	import threading, queue, multiprocessing
	ctx = multiprocessing.get_context('spawn')
	queue = ctx.Queue()
	start_exploration_processes(ctx, queue)

	try:
		replay_buffer = []
		while True:
			while not queue.empty():
				replay_buffer.extend(queue.get())
			length = len(replay_buffer)
			if length >= args().batch_size: break
		import threading
		model_lock = threading.Lock()
		start_evaluation_threads(model, model_lock, logger)
		train(replay_buffer, queue, model, model_lock, logger)
	finally:
		queue.close()

if __name__ == '__main__':
	main()
