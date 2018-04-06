import tensorflow as tf

flags = tf.app.flags

# training
flags.DEFINE_integer('batch_size', 64, 'batch_size')
flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
#flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')

# environment
flags.DEFINE_string('dataset', 'ag', 'the name of dataset')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueing examples')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary (step)')
flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving validation summary (step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model (epoch)')
flags.DEFINE_string('results', 'results', 'path for saving results')
flags.DEFINE_boolean('save', True, 'save architecture')

# Used to record architectures used -----------------------------------------------------//
# embedded layer
flags.DEFINE_integer('words', 3000, 'max number of words to extract from dataset')
flags.DEFINE_integer('length', 75, 'max length of each review (words)')
flags.DEFINE_integer('embed_dim', 50, 'vector length of embedded words')

# first conv layer
flags.DEFINE_integer('conv1_filters', 64, 'number of filters used in initial conv layer') #64
flags.DEFINE_integer('conv1_kernel', 3, 'kernel size for initial conv layer')
flags.DEFINE_integer('conv1_stride', 1, 'stride for initial conv layer')
flags.DEFINE_string('conv1_padding', 'VALID', 'padding for initial conv layer')

# first capsule layer
flags.DEFINE_integer('caps1_output', 16, 'number of capsules in the first capsule layer')
flags.DEFINE_integer('caps1_len', 8, 'vector length for first capsule layer')
flags.DEFINE_string('caps1_type', 'CONV', 'type of capsule layer [\'CONV\', \'FC\']')
flags.DEFINE_boolean('caps1_routing', False, 'boolean to use routing or not')
flags.DEFINE_integer('caps1_kernel', 2, 'kernel size (ONLY FOR CONV TYPE)')
flags.DEFINE_integer('caps1_stride', 1, 'stride (ONLY FOR CONV TYPE)')

# second capsule layer (output)
flags.DEFINE_integer('caps2_output', 4, 'number of capsules in the second capsule layer')
flags.DEFINE_integer('caps2_len', 8, 'vector length of second capsule layer')
flags.DEFINE_string('caps2_type', 'FC', 'type of capsule layer [\'CONV\', \'FC\']')
flags.DEFINE_boolean('caps2_routing', True, 'boolean to use routing or not')
#----------------------------------------------------------------------------------------//

# distributed setting
flags.DEFINE_integer('num_gpu', 1, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 4, 'number of preprocessing threads per tower')

cfg = tf.app.flags.FLAGS
