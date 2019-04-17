import os, sys, pdb, numpy as np, random, argparse, pickle, time, json
import logging, logging.config
from pprint import pprint
from collections import Counter

np.set_printoptions(precision=4)

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------    
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def debug_nn(res_list, feed_dict):
	"""
	Function for debugging Tensorflow model      

	Parameters
	----------
	res_list:       List of tensors/variables to view
	feed_dict:	Feed dict required for getting values
	
	Returns
	-------
	Returns the list of values of given tensors/variables after execution

	"""
	import tensorflow as tf
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	return res

def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def compute_kls(adj, num_nodes):
	"""
	Computes the KLS kernel for a given adjacency matrix

	Parameters
	----------
	adj:		Adjacency matrix of the graph
	num_nodes:	Number of nodes in the graph
	
	Returns
	-------
	KLS kernel for the graph
		
	"""
	min_eig = np.min(np.linalg.eigvalsh(adj))
	return (adj/np.abs(min_eig)) + np.eye(num_nodes)

def lovasz_theta(G, long_return=True, complement=False):

	(nv, edges, _) = parse_graph(G, complement)
	ne = len(edges)

	# This case needs to be handled specially.
	if nv == 1:
		return 1.0

	c = cvxopt.matrix([0.0]*ne + [1.0])
	G1 = cvxopt.spmatrix(0, [], [], (nv*nv, ne+1))
	for (k, (i, j)) in enumerate(edges):
		G1[i*nv+j, k] = 1
		G1[j*nv+i, k] = 1
	for i in range(nv):
		G1[ i*nv+i, ne] = 1

	G1 = -G1
	h1 = -cvxopt.matrix(1.0, (nv, nv))

	sol = cvxopt.solvers.sdp(c, Gs=[G1], hs=[h1])

	if long_return:
		theta = sol['x'][ne]
		Z = np.array(sol['ss'][0])
		B = np.array(sol['zs'][0])
		return { 'theta': theta, 'primal': Z, 'dual': B }
	else:
		return sol['x'][ne]


def compute_lovasz(adj, num_nodes):
	"""
	Computes the KLS kernel for a given adjacency matrix

	Parameters
	----------
	adj:		Adjacency matrix of the graph
	num_nodes:	Number of nodes in the graph
	
	Returns
	-------
	KLS kernel for the graph
		
	"""
	# lovasz = lovasz_theta(G)
	# print('\n Time to Solve Optimisation: {0} seconds'.format(time.time()-start_lovasz_theta))
	lovasz_opt = Minimising_largest_eig_val(G, max_iter, tolerance, Solver=Solv, gpu_flag=gpu_flg)
	return lovasz

def Minimising_largest_eig_val(Graph, max_iter, tolerance, Solver='SCS', Verbose=True, gpu_flag=True):
	#     Graph.remove_edges_from(G.selfloop_edges())
	nv = Graph.number_of_nodes()
	ne = Graph.number_of_edges()
	A = nx.convert_matrix.to_numpy_matrix(Graph)
	Ac = nx.to_numpy_array(nx.complement(Graph))
	# Create two scalar optimization variables.
	t = cvx.Variable((1,1))
	Y = cvx.Variable((nv,nv), PSD=True)
	# Create two constraints.
	constraints = [Y[np.where(Ac==1)] == -1, Y[np.diag_indices(nv)]== (t-1)]

	# Form objective.
	obj = cvx.Minimize(t)

	# Form and solve problem.
	prob = cvx.Problem(obj, constraints)
	if Solver == 'CVXOPT':  solution = prob.solve(solver=Solver, verbose=Verbose, max_iters=100, abstol=1e-05, reltol=1e-06, feastol=1e-07)
	if Solver == 'SCS': 	solution = prob.solve(solver=Solver, verbose=Verbose, gpu=gpu_flag, eps=tolerance, max_iters= max_iter)
	return  {'primal': np.array(Y.value,dtype=np.float64), 'theta':solution}