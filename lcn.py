from helper import *
from utils  import *
import tensorflow as tf


class LCN(object):

	def load_data(self):
		"""
		Reads the data from pickle file

		Parameters
		----------
		self.p.data: 		Name of the dataset

		Returns
		-------
		self.A:			Adjacency matrix of the graph
		self.X:			Input features of nodes
		self.num_nodes:		Number of nodes in the graph
		self.y_train:		Label information for train
		self.y_valid:		Label information for valid
		self.y_test:		Label information for test
		self.input_dim		Length of input feature vector of each node in the graph
		self.num_labels		Total number of classes to predict
		"""
		self.logger.info("loading data")
		# self.A, self.X, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = load_data(self.p.data)
		# self.X 	  	  = preprocess_features(self.X)
		# self.A  	  = preprocess_adj(self.A)

		self.data	= pickle.load(open('./data_new/data/{}.pkl'.format(self.p.data), 'rb'))[self.p.data_idx]

		self.num_nodes	= self.data['num_nodes']
		self.input_dim	= self.data['num_nodes']
		self.train_idx, \
		self.valid_idx, \
		self.test_idx 	= self.get_data_split(self.num_nodes, train_ratio=0.7, valid_ratio=0.1)
		self.num_labels	= len(np.unique(self.data['labels']))

		self.A 		= self.data['adj']
		self.X 		= sp.eye(self.num_nodes).A

		if   self.p.kernel == 'none':  	self.A 	= self.A
		elif self.p.kernel == 'kls': 	self.A	= self.data['kls']    if 'kls'    in self.data else compute_kls(self.A)
		elif self.p.kernel == 'lovasz': self.A	= self.data['lovasz'] if 'lovasz' in self.data else compute_lovasz(self.A)
		else: raise NotImplementedError


	def get_data_split(self, num_nodes, train_ratio, valid_ratio=0.0):
		idx_list  = np.arange(num_nodes)
		np.random.shuffle(idx_list)

		if valid_ratio == 0:
			num_train = int(train_ratio * num_nodes)
			return idx_list[:num_train], idx_list[num_train:]
		else:
			num_train = int(train_ratio * num_nodes)
			num_valid = int(valid_ratio * num_nodes)
			return idx_list[:num_train], idx_list[num_train: num_train+num_valid], idx_list[num_train+num_valid:]


	def add_placehoders(self):
		"""
		Defines the placeholder required for the model
		"""
		self.features		= tf.placeholder(tf.float32, 		shape=[self.num_nodes, self.input_dim], name='features')
		self.adj_mat		= tf.placeholder(tf.float32, 		shape=[self.num_nodes, self.num_nodes], name='support')
		self.labels		= tf.placeholder(tf.int32, 	  	shape=[None], 				name='labels')
		self.split_idx		= tf.placeholder(tf.int32, 							name='split_idx')
		self.dropout		= tf.placeholder_with_default(0.,   	shape=(), 				name='dropout')
		self.num_nonzero	= tf.placeholder(tf.int32, 							name='num_nonzero')

	def create_feed_dict(self, split='train'):
		"""
		Creates a feed dictionary for the batch

		Parameters
		----------
		split:		Indicates the split of the data - train/valid/test

		Returns
		-------
		feed_dict	Feed dictionary to be fed during sess.run
		"""
		feed = {}

		feed[self.features] 			= self.X
		feed[self.adj_mat] 			= self.A
		feed[self.num_nonzero] 			= self.num_nodes
		feed[self.labels] 			= self.data['labels']

		if split == 'train': 	feed[self.split_idx] = self.train_idx
		elif split == 'test': 	feed[self.split_idx] = self.test_idx
		else: 			feed[self.split_idx] = self.valid_idx

		return feed


	def sparse_dropout(self, x, keep_prob, noise_shape):
		"""
		Dropout for sparse tensors.
		"""
		random_tensor  = keep_prob
		random_tensor += tf.random_uniform(noise_shape)
		dropout_mask   = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
		pre_out        = tf.sparse_retain(x, dropout_mask)
		return pre_out * (1./keep_prob)


	def glorot(self, shape, name=None):
		"""
		Glorot & Bengio (AISTATS 2010) init.
		"""
		init_range = np.sqrt(6.0/(shape[0]+shape[1]))
		initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
		return tf.Variable(initial, name=name)

	def GCNLayer(self, gcn_in, adj_mat, input_dim, output_dim, act, dropout, num_nonzero, input_sparse=False, name='GCN'):
		"""
		GCN Layer Implementation

		Parameters
		----------
		gcn_in:		Input to GCN Layer
		adj_mat:	Adjacency matrix of the graph
		input_dim:	Dimension of input to GCN Layer 
		output_dim:	Dimension of the output of GCN layer
		act:		Activation to be used at the end
		dropout:	Dropout used
		num_nonzeros:	self.X[1].shape
		name 		Name of the layer (used for creating variables, keep it different for different layers)

		Returns
		-------
		out		Output of GCN Layer
		"""

		with tf.name_scope(name):
			with tf.variable_scope('{}_vars'.format(name)) as scope:
				wts  = tf.get_variable('weights', [input_dim, output_dim], 	 initializer=tf.initializers.glorot_normal())
				bias = tf.get_variable('bias', 	  [output_dim], 	   	 initializer=tf.initializers.glorot_normal())
				self.l2_var.extend([wts, bias])
			
			if input_sparse:
				gcn_in  = self.sparse_dropout(gcn_in, 1 - dropout, num_nonzero)
				pre_sup = tf.sparse_tensor_dense_matmul(gcn_in,  wts)
			else:
				gcn_in = tf.nn.dropout(gcn_in, 1-dropout)
				pre_sup = tf.matmul(gcn_in,  wts)

			# support = tf.sparse_tensor_dense_matmul(adj_mat, pre_sup)
			support = tf.matmul(adj_mat, pre_sup)

		return act(support)

	def add_model(self):
		"""
		Creates the Computational Graph

		Parameters
		----------

		Returns
		-------
		nn_out:		Logits for each node in the graph
		"""

		gcn1_out = self.GCNLayer(		
				gcn_in 			= self.features,
				adj_mat 		= self.adj_mat,
				input_dim		= self.input_dim,
				output_dim		= self.p.gcn_dim,
				act			= tf.nn.relu,
				dropout			= self.dropout,
				num_nonzero 		= self.num_nonzero,
				input_sparse 		= False,
				name 			= 'GCN_1'
			)
		
		gcn2_out = self.GCNLayer(		
				gcn_in 			= gcn1_out,
				adj_mat 		= self.adj_mat,
				input_dim		= self.p.gcn_dim,
				output_dim		= self.num_labels,
				act			= lambda x: x,
				dropout			= self.dropout,
				num_nonzero 		= self.num_nonzero,
				input_sparse 		= False,
				name 			= 'GCN_2'
			)

		nn_out = gcn2_out
		return nn_out

	def get_accuracy(self, nn_out):
		pred			= tf.gather(nn_out, self.split_idx)
		labels			= tf.gather(self.labels, self.split_idx)
		correct_prediction	= tf.equal(tf.cast(tf.argmax(pred, 1), tf.int32), labels)		# Identity position where prediction matches labels
		accuracy 		= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))				# Cast result to float

		return accuracy


	def add_loss_op(self, nn_out):
		"""
		Computes loss based on logits and actual labels

		Parameters
		----------
		nn_out:		Logits for each bag in the batch

		Returns
		-------
		loss:		Computes loss based on prediction and actual labels of the bags
		"""

		pred	= tf.gather(nn_out, self.split_idx)
		labels	= tf.gather(self.labels, self.split_idx)
		loss	= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels)) 		# Compute cross entropy loss

		for var in self.l2_var:
			loss += self.p.l2 * tf.nn.l2_loss(var)

		return loss

	def add_optimizer(self, loss, isAdam=True):
		"""
		Add optimizer for training variables

		Parameters
		----------
		loss:		Computed loss

		Returns
		-------
		train_op:	Training optimizer
		"""
		with tf.name_scope('Optimizer'):
			if isAdam:  optimizer = tf.train.AdamOptimizer(self.p.lr)
			else:       optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)

		return train_op

	def __init__(self, params):
		"""
		Constructor for the main function. Loads data and creates computation graph. 

		Parameters
		----------
		params:		Hyperparameters of the model

		Returns
		-------
		"""
		self.p  = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p)); pprint(vars(self.p))

		self.l2_var = []

		self.load_data()
		self.add_placehoders()

		nn_out    	= self.add_model()
		self.loss 	= self.add_loss_op(nn_out)
		self.accuracy 	= self.get_accuracy(nn_out)

		self.train_op = self.add_optimizer(self.loss)
		self.cost_val = []

		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None


	def evaluate(self, sess, split='valid'):
		"""
		Performs evaluation on the dataset

		Parameters
		----------
		sess:		Session of tensorflow
		split:		Dataset split to evaluate on

		Returns
		-------
		loss:		Loss over the dataset split
		acc:		Accuracy on the data
		time:		Time taken for entire evaluation
		"""
		t_test 		= time.time()							# Measuring time
		feed_dict 	= self.create_feed_dict(split=split)  				# Defines the feed_dict to be fed to NN
		loss, acc 	= sess.run([self.loss, self.accuracy], feed_dict=feed_dict) 	# Computer loss and accuracy
		return loss, acc, (time.time() - t_test)				# return loss, accuracy and time taken

	def run_epoch(self, sess, epoch, shuffle=True):
		"""
		Runs one epoch of training

		Parameters
		----------
		sess:		Session of tensorflow
		data:		Data to train on
		epoch:		Epoch number
		shuffle:	Shuffle data while before creates batches

		Returns
		-------
		"""

		t = time.time()
		feed_dict = self.create_feed_dict(split='train')
		feed_dict.update({self.dropout: self.p.dropout})

		# Training step
		_, train_loss, train_acc = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)

		# Validation
		val_loss, val_acc, duration = self.evaluate(sess, split='valid')
		self.cost_val.append(val_loss)

		if val_acc > self.best_val_acc: 
			self.best_val_acc = val_acc
			_, self.best_test_acc, _ = self.evaluate(sess, split='test')

		self.logger.info('E: {}, Train Loss: {:.3}, Train Acc: {:3}, Val Loss: {:.3}, Val Acc: {:.3}'.format(epoch, train_loss, train_acc, val_loss, val_acc))


	def fit(self, sess):
		"""
		Trains the model and finally evaluates on test

		Parameters
		----------
		sess:		Tensorflow session object

		Returns
		-------
		"""
		self.summ_writer = tf.summary.FileWriter("tf_board/GCN_WORD/" + self.p.name, sess.graph)
		self.saver     = tf.train.Saver()
		save_dir  = 'checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		self.save_path = os.path.join(save_dir, 'best_int_avg')

		self.best_val_acc  = 0.0
		self.best_test_acc = 0.0

		if self.p.restore:
			self.saver.restore(sess, self.save_path)

		for epoch in range(self.p.max_epochs):
			train_loss = self.run_epoch(sess, epoch)

		self.logger.info('Best Valid: {}, Best Test: {}'.format(self.best_val_acc, self.best_test_acc))


if __name__== "__main__":

	parser = argparse.ArgumentParser(description='WORD GCN')

	parser.add_argument('-data',     dest="data",    	default='1000_644', 		help='Dataset to use')
	parser.add_argument('-data_idx', dest="data_idx",    	default=0, 	type=int,	help='Dataset to use')
	parser.add_argument('-gpu',      dest="gpu",            default='0',                	help='GPU to use')
	parser.add_argument('-name',     dest="name",           default='test',             	help='Name of the run')
	parser.add_argument('-kernel',   dest="kernel",         default='kls',                  help='Kernel name', choices=['none', 'kls', 'lovasz'])
	parser.add_argument('-lr',       dest="lr",             default=0.01,   type=float,     help='Learning rate')
	parser.add_argument('-epoch',    dest="max_epochs",     default=200,    type=int,       help='Max epochs')
	parser.add_argument('-l2',       dest="l2",             default=5e-4,   type=float,     help='L2 regularization')
	parser.add_argument('-seed',     dest="seed",           default=1234,   type=int,       help='Seed for randomization')
	parser.add_argument('-gcn_dim',  dest="gcn_dim",     	default=16,     type=int,       help='GCN hidden dimension')
	parser.add_argument('-drop',     dest="dropout",        default=0.5,    type=float,     help='Dropout for full connected layer')
	parser.add_argument('-opt',      dest="opt",            default='adam',             	help='Optimizer to use for training')
	parser.add_argument('-dump',  	 dest="dump",       	action='store_true',        	help='Dump context and embed matrix')
	parser.add_argument('-restore',  dest="restore",        action='store_true',        	help='Restore from the previous best saved model')
	parser.add_argument('-logdir',   dest="log_dir",        default='./log/',       	help='Log directory')
	parser.add_argument('-config',   dest="config_dir",     default='./config/',       	help='Config directory')

	args = parser.parse_args()

	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)

	model = LCN(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)

	print('Model Trained Successfully!!')