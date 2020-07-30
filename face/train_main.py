from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from classifier import training
import os

def train_again(datadir=os.path.join(os.getcwd(),'face','training_files','face'),modeldir=os.path.join(os.getcwd(),'face','model','20180402-114759.pb'),
	classifier_filename=os.path.join(os.getcwd(),'face','class','classifier.pkl')):

	print ("Training Start")
	obj=training(datadir,modeldir,classifier_filename)
	get_file=obj.main_train()
	print('Saved classifier model to file "%s"' % get_file)


if __name__=='__main__':
	train_again()