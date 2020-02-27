import warnings
warnings.filterwarnings("ignore")
from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork
import cv2
import numpy as np
import os
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import sys

def read_video(video_addr):

	## MAKE IT A YIELDER OF BATHCES OF RANDOM(90,100) FRAMES EACH TILL THE END OF THE VIDEO UTILIZING ALL THE INFORMATION

	video_capture = cv2.VideoCapture(video_addr)
	_,imgs=video_capture.read()
	imgs=np.reshape(cv2.resize(imgs,(299,299)),(1,299,299,3))
	while True:
		ret, frame = video_capture.read()
		if ret:
			frame=np.reshape(cv2.resize(frame,(299,299)),(1,299,299,3))
		else:
			break
		imgs=np.vstack((imgs,frame))
	
	if imgs.shape[0]>100:
		imgs=imgs[10:100]
	print(imgs.shape)
	return imgs

class everything:
	def __init__(self,train_input_datadir,train_output_datadir,test_input_datadir=None,test_output_datadir=None):
		self.in_dir=train_input_datadir
		self.out_dir=train_output_datadir

		self.net_pose=HumanPoseIRNetwork()
		self.net_gait=GaitNetwork(recurrent_unit = 'GRU', rnn_layers = 2)
		self.net_pose.restore(os.path.join(os.getcwd(),'models','Human3.6m.ckpt'))
		self.net_gait.restore(os.path.join(os.getcwd(),'models','H3.6m-GRU-1.ckpt'))

		self.path_to_test_videos=test_input_datadir
		self.path_to_test_frames=test_output_datadir

		self.label_filename=os.path.join(os.getcwd(),'saved_files','labelencoder.pkl')
		self.classifier_filename=os.path.join(os.getcwd(),'saved_files','classifier.pkl')

	def check_frames(self,in_dir,out_dir):
		if not os.path.exists(in_dir):
			print("Data Directory Not Found  :: Exiting")
			sys.exit()

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		for x in os.listdir(in_dir):
			if not os.path.exists(os.path.join(out_dir,x)):
				os.makedirs(os.path.join(out_dir,x))

		train_features_in_out_list=[]

		for y in os.listdir(in_dir):
			for x in os.listdir(os.path.join(in_dir,y)):
				x_=x[:-3]+'npy'
				if not os.path.exists(os.path.join(out_dir,y,x_)):
					train_features_in_out_list.append([os.path.join(in_dir,y,x),os.path.join(out_dir,y,x_)])

		print("{} NEW FEATURES NEEDS CONVERSION".format(len(train_features_in_out_list)))

		return train_features_in_out_list

	def convert(self,train_features_in_out_list):
		print("STARTING CONVERSION")
		count=len(train_features_in_out_list)
		for x,y in train_features_in_out_list:
			try:	
				spatial_features=self.net_pose.feed_forward_features(read_video(x))
				np.save(y,spatial_features)
			except:
				print("ERROR IN CONVERTING TO SPATIAL FEATURES :: SKIPPING -- {}".format(x))
		
		print("MADE {} NEW FEATURES :: SKIPPED {} VIDEOS ".format(count,len(train_features_in_out_list)-count))

	def make_feature_label(self,out_dir,for_train=True):
		print('MAKING DATA')
		feature_label=[]
		for x in os.listdir(out_dir):
			for y in os.listdir(os.path.join(out_dir,x)):
				frame=np.load(os.path.join(out_dir,x,y))
				features,_=self.net_gait.feed_forward(frame)
				feature_label.append([features,x])

		random.shuffle(feature_label)
		features=[]
		label=[]
		for x,y in feature_label:
			features.append(x)
			label.append(y)
		features=np.asarray(features)
		label=np.asarray(label)
		return features,label


	def train(self):
		train_features_in_out_list=self.check_frames(self.in_dir,self.out_dir)
		if len(train_features_in_out_list)>0:
			self.convert(train_features_in_out_list)

		features,label=self.make_feature_label(self.out_dir)
		print('TRAINING STARTED')
		le=LabelEncoder()
		label=le.fit_transform(label)

		train_feature,val_feature,train_label,val_label=train_test_split(features,label,test_size=0.1,random_state=0)
		print('Training on {} rows   AND    validation on {} rows'.format(np.shape(train_feature)[0],np.shape(val_feature)[0]))
		model = SVC(kernel='linear', probability=True)
		model.fit(train_feature, train_label)
		print('TRAINING DONE')
		print('Accuracy on training set --\t{}'.format(model.score(train_feature,train_label)*100))
		print('Accuracy on validation set --\t{}'.format(model.score(val_feature,val_label)*100))
		
		with open(self.label_filename, 'wb+') as outfile:
			pickle.dump(le, outfile)
		print('SAVED LABELENCODER ')
		
		with open(self.classifier_filename, 'wb+') as outfile:
			pickle.dump(model, outfile)
		print('SAVED CLASSIFIER ')


	def test(self):
		test_feature_in_out_list=self.check_frames(self.path_to_test_videos,self.path_to_test_frames)
		if len(test_feature_in_out_list)>0:
			self.convert(test_feature_in_out_list)

		with open(self.label_filename, 'rb') as outfile:
			le=pickle.load(outfile)

		with open(self.classifier_filename, 'rb') as outfile:
			model=pickle.load(outfile)

		features,label=self.make_feature_label(self.path_to_test_frames)
		output=model.predict(features)
		label=le.transform(label)

		print("Prediction on {} features ".format(label.shape))
		a=np.stack((le.inverse_transform(output).reshape([-1]),le.inverse_transform(label).reshape([-1])),axis=1)

		print("Individual Classes Prediction")
		print(a)

		accuracy=(output==label).mean()

		print('TESTING ACCURACY :: {}'.format(accuracy*100))



if __name__=='__main__':
	train_inp_=os.path.join(os.getcwd(),'training_files','videos')
	train_out_=os.path.join(os.getcwd(),'training_files','frames')
	test_inp_=os.path.join(os.getcwd(),'testing_files','videos')
	test_out_=os.path.join(os.getcwd(),'testing_files','frames')

	pr=everything(train_inp_,train_out_,test_inp_,test_out_)
	
	# model=RandomForestClassifier()
	is_train=True
	do_test=False

	if is_train:
		pr.train()
	if do_test:
		pr.test()
