import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from config import cfg
from utils import load_imdb, record
from model import CapsNet

def save_to():
	if not os.path.exists(cfg.results):
		os.mkdir(cfg.results)

	if cfg.is_training:
		loss = cfg.results + '/loss.csv'
		train_acc = cfg.results + '/train_acc.csv'
		val_acc = cfg.results + '/val_acc.csv'

		if os.path.exists(loss):
			os.remove(loss)
		if os.path.exists(train_acc):
			os.remove(train_acc)
		if os.path.exists(val_acc):
			os.remove(val_acc)

		f_loss = open(loss, 'w')
		f_loss.write('step, loss\n')

		f_train_acc = open(train_acc, 'w')
		f_train_acc.write('step, train_acc\n')

		f_val_acc = open(val_acc, 'w')
		f_val_acc.write('step, val_acc\n')

		return f_loss, f_train_acc, f_val_acc

	else:
		test_acc = cfg.results + '/test_acc.csv'

		if os.path.exists(test_acc):
			os.remove(test_acc)

		f_test_acc = open(test_acc, 'w')
		f_test_acc.write('test_acc\n')
		return f_test_acc


def train(model, supervisor, num_label):
	losses = []
	accs = []

	trX, trY, num_tr_batch, valX, valY, num_val_batch = load_imdb(cfg.batch_size, cfg.words, cfg.length, is_training=True)

	f_loss, f_train_acc, f_val_acc = save_to()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with supervisor.managed_session(config=config) as sess:
		print('\nSupervisor Prepared')

		for epoch in range(cfg.epoch):
			print('Training for epoch ' + str(epoch+1) + '/' + str(cfg.epoch) + ':')

			if supervisor.should_stop():
				print('Supervisor stopped')
				break

			for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
				start = step * cfg.batch_size
				end = start + cfg.batch_size
				global_step = epoch * num_tr_batch + step

				if global_step % cfg.train_sum_freq == 0:
					_, loss, train_acc = sess.run([model.train_op, model.total_loss, model.accuracy])

					losses.append(loss)
					accs.append(train_acc)

					#======================================
					print(loss, train_acc)
					#======================================
					#v_j= sess.run(model.v_j)
					#print(v_j)
					#======================================
					assert not np.isnan(loss), 'loss is nan...'
					# Need to add summary strings
					#supervisor.summary_writer.add_summary(summary_str, global_step)

					f_loss.write(str(global_step) + ',' + str(loss) + '\n')
					f_loss.flush()
					f_train_acc.write(str(global_step) + ',' + str(train_acc) + '\n')
					f_train_acc.flush()

				else:
					sess.run(model.train_op)

				if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
					val_acc = 0
					for i in range(num_val_batch):
						start = i * cfg.batch_size
						end = start + cfg.batch_size
						acc = sess.run(model.accuracy, {model.X: valX[start:end], model.Y: valY[start:end]})
						val_acc += acc
					val_acc = val_acc / num_val_batch
					f_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
					f_val_acc.flush()

			if (epoch + 1) % cfg.save_freq == 0:
				supervisor.saver.save(sess, cfg.logdir + '/model_epoch_{0:.4g}_step_{1:.2g}'.format(epoch, global_step))

		supervisor.saver.save(sess, cfg.logdir + '/model_epoch_{0:.4g}_step_{1:.2g}'.format(epoch, global_step))


		f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
		ax1.plot(losses)
		ax2.plot(accs)
		plt.show()

		f_loss.close()
		f_train_acc.close()
		f_val_acc.close()

		return losses[-1], accs[-1]


def evaluation(model, supervisor, num_label):
	teX, teY, num_te_batch = load_imdb(cfg.batch_size, cfg.words, cfg.length, is_training=False)

	cfg.is_training = False
	f_test_acc = save_to()

	with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
		tf.logging.info('Model restored...')

		test_acc = 0

		for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
			start = i * cfg.batch_size
			end = start + cfg.batch_size
			acc = sess.run(model.accuracy, {model.X: teX[start:end], model.Y: teY[start:end]})
			test_acc += acc
		test_acc = test_acc / num_te_batch
		f_test_acc.write(str(test_acc))
		f_test_acc.close()
		print('Test accuracy saved to ' + cfg.results + '/test_acc.csv')
		print('Test accuracy:', test_acc)

	return test_acc


def main(_):
	tf.logging.info('Loading Graph...')
	num_label = 1
	model = CapsNet()
	tf.logging.info('Graph loaded')

	sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

	if not cfg.is_training:
		_ = evaluation(model, sv, num_label)

	else:
		tf.logging.info('Start is_training...')
		loss, acc = train(model, sv, num_label)
		tf.logging.info('Training done')
		test_acc = evaluation(model, sv, num_label)
		record(loss, acc, test_acc)


if __name__ == '__main__':
	tf.app.run()