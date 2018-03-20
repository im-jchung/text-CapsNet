import os
import datetime
import numpy as np
import tensorflow as tf

from config import cfg
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.preprocessing import sequence

def load_imdb(batch_size, words, length, is_training=True):
	(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=words)

	X_train = sequence.pad_sequences(X_train, maxlen=length)
	X_test = sequence.pad_sequences(X_test, maxlen=length)

	y_train_ohe = to_categorical(y_train)
	y_test_ohe = to_categorical(y_test)

	y_train = y_train_ohe
	y_test = y_test_ohe

	if is_training:
		trX = X_train[:22500]
		trY = y_train[:22500]

		valX = X_train[22500:,]
		valY = y_train[22500:]

		num_tr_batch = 22500 // batch_size
		num_val_batch = 2500 // batch_size

		return trX, trY, num_tr_batch, valX, valY, num_val_batch

	else:
		num_te_batch = 25000 // batch_size
		return X_test, y_test, num_te_batch

def get_batch_dataset(dataset, batch_size, words, length, num_threads):
	if dataset == 'imdb':
		trX, trY, num_tr_batch, valX, valY, num_val_batch = load_imdb(batch_size, words, length, is_training=True)
	data_queues = tf.train.slice_input_producer([trX, trY])
	X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
								  batch_size=batch_size,
								  capacity=batch_size*64,
								  min_after_dequeue=batch_size*32,
								  allow_smaller_final_batch=False)
	return X,Y


def record(loss, acc, test_acc):
	archs = cfg.results + '/archs.txt'
	if not os.path.exists(cfg.results):
		os.mkdir(cfg.results)

	now = datetime.datetime.now()
	now = str(now)[0:-7]

	f = open(archs, 'a+')
	start = '=' * 50 + ' ' + now + ' ' + '=' * 50
	f.write(start + '\n')

	Epochs    = '{0: <20}'.format(str.format('Epochs: {}', cfg.epoch))
	Words     = '{0: <20}'.format(str.format('Words: {}', cfg.words))
	Length    = '{0: <20}'.format(str.format('Length: {}', cfg.length))
	Routing   = '{0: <20}'.format(str.format('Routing: {}', cfg.iter_routing))

	embed_dim = '{0: <20}'.format(str.format('embed_dim: {}', cfg.embed_dim))

	conv1_filters = '{0: <20}'.format(str.format('filters: {}', cfg.conv1_filters))
	conv1_kernel  = '{0: <20}'.format(str.format('kernel: {}', cfg.conv1_kernel))
	conv1_stride  = '{0: <20}'.format(str.format('stride: {}', cfg.conv1_stride))
	conv1_pad     = '{0: <20}'.format(str.format('padding: {}', cfg.conv1_padding))

	caps1_output  = '{0: <20}'.format(str.format('num_output: {}', cfg.caps1_output))
	caps1_len     = '{0: <20}'.format(str.format('vec_len: {}', cfg.caps1_len))
	caps1_type    = '{0: <20}'.format(str.format('type: {}', cfg.caps1_type))
	caps1_routing = '{0: <20}'.format(str.format('routing: {}', cfg.caps1_routing))
	caps1_kernel  = '{0: <20}'.format(str.format('kernel: {}', cfg.caps1_kernel))
	caps1_stride  = '{0: <20}'.format(str.format('stride: {}', cfg.caps1_stride))

	caps2_output  = '{0: <20}'.format(str.format('num_output: {}', cfg.caps2_output))
	caps2_len     = '{0: <20}'.format(str.format('vec_len: {}', cfg.caps2_len))
	caps2_type    = '{0: <20}'.format(str.format('type: {}', cfg.caps2_type))
	caps2_routing = '{0: <20}'.format(str.format('routing: {}', cfg.caps2_routing))

	loss       = '{0: <20}'.format(str.format('Loss: {:.3f}', loss))
	acc        = '{0: <20}'.format(str.format('Acc: {:.2f}', acc))
	test_acc  = '{0: <20}'.format(str.format('Test Acc: {:.4f}', test_acc))

	gen   = 'Gen.  | ' + Epochs + Words + Length + Routing + '\n'
	embed = 'Embed | ' + embed_dim + '\n'
	conv1 = 'Conv1 | ' + conv1_filters + conv1_kernel + conv1_stride + conv1_pad + '\n'
	caps1 = 'Caps1 | ' + caps1_output + caps1_len + caps1_type + caps1_routing + caps1_kernel + caps1_stride + '\n'
	caps2 = 'Caps2 | ' + caps2_output + caps2_len + caps2_type + caps2_routing + '\n'
	out   = 'Res.  | ' + loss + acc + test_acc + '\n'

	content = gen + embed + conv1 + caps1 + caps2 + out

	f.write(content)

	end = '=' * 58 + ' end ' + '=' * 58 + '\n'
	f.write(end + '\n')

	f.close()

	print('Architecture and results saved to:', archs)