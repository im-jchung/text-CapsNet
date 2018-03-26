import tensorflow as tf

from utils import get_batch_dataset
from capsLayer import CapsLayer
from config import cfg

epsilon = 1e-9

class CapsNet(object):

	def __init__(self, is_training=True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			if is_training:
				self.X, self.Y = get_batch_dataset(cfg.dataset, cfg.batch_size, cfg.words, cfg.length, cfg.num_threads)

				self.build_arch()
				self.loss()
				self.summary_()

				self.global_step = tf.Variable(0, name='global_step', trainable=False)
				self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
				self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
			else:
				self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, cfg.length))
				self.Y = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
				self.build_arch()

		tf.logging.info('Setting up...')

	def build_arch(self):
		with tf.variable_scope('Embedding'):
			embed = tf.contrib.layers.embed_sequence(self.X, vocab_size=cfg.words, embed_dim=cfg.embed_dim)

		with tf.variable_scope('Conv1_layer'):
			conv1 = tf.layers.conv1d(embed, filters=cfg.conv1_filters, kernel_size=cfg.conv1_kernel, strides=cfg.conv1_stride, padding=cfg.conv1_padding)

		with tf.variable_scope('First_caps_layer'):
			firstCaps = CapsLayer(num_outputs=cfg.caps1_output, vec_len=cfg.caps1_len, layer_type=cfg.caps1_type, with_routing=cfg.caps1_routing)
			caps1 = firstCaps(conv1, kernel_size=cfg.caps1_kernel, stride=cfg.caps1_stride)

		with tf.variable_scope('Second_caps_layer'):
			secondCaps = CapsLayer(num_outputs=cfg.caps2_output, vec_len=cfg.caps2_len, layer_type=cfg.caps2_type, with_routing=cfg.caps2_routing)
			self.caps2 = secondCaps(caps1, kernel_size=None, stride=None)

		#========================================

		with tf.variable_scope('Out'):
			self.v_j = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)

		#========================================

		# Add layers as needed

		# Decoder network? IDK how we can do this for linguistics

	def loss(self):
		# We can use binary cross entropy, I believe
		# p * -tf.log(q) + (1 - p) * -tf.log(1 - q)
		#L_k = tf.reduce_sum(self.v_j, axis=[1,2,3])
		logits = tf.squeeze(self.v_j)
		self.Y = tf.cast(self.Y, tf.float32)
		self.total_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.Y)
		#self.total_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=logits)
		self.total_loss = tf.reduce_mean(self.total_loss)

	def summary_(self):
		train_summary = []
		train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))

		img = tf.reshape(self.caps2, shape=(cfg.batch_size, 4, 16, 1))
		train_summary.append(tf.summary.image('image', img))
		self.train_summary = tf.summary.merge(train_summary)


		preds = tf.round(tf.squeeze(self.v_j))
		preds = tf.cast(preds, tf.int32)

		correct_prediction = tf.equal(tf.to_int32(self.Y), preds)
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))