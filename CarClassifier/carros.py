import tensorflow as tf
import numpy
import numpy as np
import math
from PIL import Image
from six.moves import xrange  # pylint: disable=redefined-builtin
import os

path, dirs, files = next(os.walk("d:/Documents/code/Carga_python_2/train/testCarros"))
file_count = len(files)
print("Cantidad de imagenes: ", file_count)

learning_rate = 0.01
num_test = int(file_count)

IMAGE_WIDTH  = 80
IMAGE_HEIGHT = 40
IMAGE_DEPTH  = 1
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
NUM_CLASSES  = 2
BATCH_SIZE    = 100

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

def read_my_list2( minId, maxId, folder ):
	""" create list with train/no and train/go from 1 to maxid
		max maxId = 50000
		"""
	
	filenames = []
	labels   = []
	#labels = np.zeros( ( ( maxId - minId ) * 2, 2 ) )
	for num in range( minId, maxId+1 ):
		
		#Conjunto de Prueba
		##filenames.append( "d:/Documents/code/Imagenes/" + folder + "/5_imgs/" + name_si( num ) + ".jpg" )
		filenames.append( "d:/Documents/code/Carga_python_2/" + folder + "/testCarros/" + name_no( num ) + ".jpg" )
		#labels[ ( ( num - minId ) * 2 ) + 1 ][ 0 ] = 1
		labels.append( int( 0 ) )
		#labels=[1,0,1,1,0,1]
		
		#print( num_name(num) )

	#labels=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

	# return list with all filenames
	print( "number of labels2: " + str( len( labels ) ) )
	print( "number of images2: " + str( len( filenames ) ) )
	return filenames, labels

def num_name( id ):
	
	# create string where id = 5 becomes 00005
	
	ret = str( id )
	while ( len( ret ) < 5 ):
		ret = "0" + ret;
	
	return ret;

def name_si( id ):
	
	ret = str( id )
	ret = "im" + ret;
	
	return ret;

def name_no( id ):
	
	ret = str( id )
	ret = "im" + ret;
	
	return ret;

def read_images_from_disk(input_queue):
	"""Consumes a single filename and label as a ' '-delimited string.
		Args:
		filename_and_label_tensor: A scalar string tensor.
		Returns:
		Two tensors: the decoded image, and the string label.
		"""
	label = input_queue[1]
	print( "read file "  )
	file_contents = tf.read_file(input_queue[0])
	example = tf.image.decode_jpeg( file_contents, channels = 1 )
	example = tf.image.resize_images(images=example,size=[IMAGE_HEIGHT,IMAGE_WIDTH])
	example = tf.reshape( example, [ IMAGE_PIXELS ] )
	example.set_shape( [ IMAGE_PIXELS ] )
	
	example = tf.cast( example, tf.float32 )
	example = tf.cast( example, tf.float32 ) * ( 1. / 255 ) - 0.5
	
	label = tf.cast( label, tf.int64 )
	
	label = tf.one_hot( label, 2, 0, 1 )
	label = tf.cast( label, tf.float32 )
	
	print( "file read " )
	return  example, label

def fill_feed_dict(image_batch, label_batch, imgs, lbls):
  feed_dict = {
	  imgs: image_batch,
	  lbls: label_batch,
  }
  return feed_dict

# tf Graph input
x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)


def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
	
	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1, k=2)
	
	# Convolution Layer
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2, k=2)
	
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	
	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

	#out=softmax(out,dim=-1,name=None)
	return out

weights = {
	# 5x5 conv, 1 input, 32 outputs
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]),name='wc1'),
	# 5x5 conv, 32 inputs, 64 outputs
	'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]),name='wc2'),
	# fully connected, 7*7*64 inputs, 1024 outputs
	'wd1': tf.Variable(tf.random_normal([int(IMAGE_WIDTH/4)*int(IMAGE_HEIGHT/4)*64, 1024]),name='wd1'),
	# 1024 inputs, 10 outputs (class prediction)
	'out': tf.Variable(tf.random_normal([1024, NUM_CLASSES]),name='out1')
}

biases = {
	'bc1': tf.Variable(tf.random_normal([32]),name='bc1'),
	'bc2': tf.Variable(tf.random_normal([64]),name='bc2'),
	'bd1': tf.Variable(tf.random_normal([1024]),name='bd1'),
	'out': tf.Variable(tf.random_normal([NUM_CLASSES]),name='out2')
}

# Construct model
y = conv_net(x, weights, biases, keep_prob)
softmax = tf.exp(y) / tf.reduce_sum(tf.exp(y))

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
prediction=tf.argmax(y,1)
salida = tf.argmax(y, 1)

# Evaluate model
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# MI PRUEBA
# get filelist and labels for tESTING
image_list_test2, label_list_test2 = read_my_list2( 1, file_count, "train" )

# create queue for training
input_queue_test2 = tf.train.slice_input_producer( [ image_list_test2, label_list_test2 ],shuffle=False)

# read files for training
image_test2, label_test2 = read_images_from_disk( input_queue_test2 )

# read from the input queue.
image_batch_test2, label_batch_test2 = tf.train.batch( [ image_test2, label_test2 ], batch_size = num_test )

saver=tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, './CNN-CarrosTaxis_Neg')
	#print('weight:',sess.run(weights))
	#print('biases:',sess.run(biases))
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	imgs_test2, lbls_test2 = sess.run([image_batch_test2, label_batch_test2])
	#salida=sess.run(y,feed_dict={x: imgs_test2 })
	print (" ")
	print (" Resultados de Testing:")
	print (" ")
	#print(salida)
	#print (prediction.eval(feed_dict={x: imgs_test2 ,keep_prob: dropout},session=sess))
	#print (" ")
	#print ("Correct prediction: ", correct_pred.eval(feed_dict={x: imgs_test2 , y_: lbls_test2, keep_prob: dropout}))
	#softmax = tf.exp(y) / tf.reduce_sum(tf.exp(y))
	#print("Softmax: ",softmax)
	#print ("Softmax2: ", softmax.eval(feed_dict={x: imgs_test2 , y_: lbls_test2, keep_prob: dropout}))
	'''
	print ("Accuracy Prueba: ", accuracy.eval(feed_dict={x: imgs_test2 , y_: lbls_test2, keep_prob: dropout}))
	print (" ")
	print ("Correct prediction: ", correct_pred.eval(feed_dict={x: imgs_test2 , y_: lbls_test2, keep_prob: dropout}))
	print (" ")
	print ("label_list_test: ",label_list_test2)
	print (" ")
	'''
	u=prediction.eval(feed_dict={x: imgs_test2 , keep_prob: dropout}, session=sess)
	print (" ")
	print ("predictions: ", u)
	print (" ")
	#print ("label_list_test: ",label_list_test2)
	#print (" ")

	file = open('testCarros.txt', 'w')

	for item in u:
		file.write("%s " % item)

	coord.request_stop()
	coord.join(threads)