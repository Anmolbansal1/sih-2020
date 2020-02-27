from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import sys
import shutil
from datetime import datetime
from train_main import train_again
from data_preprocess import img_to_face
from identify_webcam import test

import warnings
warnings.filterwarnings('ignore')


class face_model:
	def __init__(self):
		self.modeldir = os.path.join(os.getcwd(),'face','model','20170511-185253.pb')
		self.classifier_filename = os.path.join(os.getcwd(),'face','class','classifier.pkl')
		self.npy=os.path.join(os.getcwd(),'face','npy')
		self.MY_FACE_DIRECTORY=os.path.join(os.getcwd(),'face','training_files','face')
		self.MY_IMG_DIRECTORY=os.path.join(os.getcwd(),'face','training_files','img')
		self.TRAIN_FRAMES=[]
		### DEBUGGER
		self.counter=0

		self.classes=os.listdir(os.path.join(os.getcwd(),'face','training_files','img'))

		# constructor takes in list of classes
		self.save_name='out.avi'
		self.classes.sort()
		self.classes=['Unknown']+self.classes
		self.prediction_class=np.zeros((len(self.classes)+1))
		self.num_imgs=0
		with tf.Graph().as_default():
			self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
			self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))
			with self.sess.as_default():
				self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, self.npy)

				self.minsize = 20  # minimum size of face
				self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
				self.factor = 0.709  # scale factor
				self.image_size = 182
				self.input_image_size = 160

				print('Loading feature extraction model')
				facenet.load_model(self.modeldir)

				self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
				self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
				self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
				self.embedding_size = self.embeddings.get_shape()[1]


				self.classifier_filename_exp = os.path.expanduser(self.classifier_filename)
				with open(self.classifier_filename_exp, 'rb') as infile:
				    (self.model, self.class_names) = pickle.load(infile)

				self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
				self.writer = None
				print('Start Recognition!')
				self.start_rec=False

	def predict(self,frame):

		frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

		if frame.ndim == 2:
			frame = facenet.to_rgb(frame)
		frame = frame[:, :, 0:3]
		bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
		nrof_faces = bounding_boxes.shape[0]
		# print('Face Detected: %d' % nrof_faces)

		if self.start_rec:
			if self.writer is None:
				(h, w) = frame.shape[:2]
				self.writer = cv2.VideoWriter(self.save_name, self.fourcc, 10,(w, h), True)
			self.counter+=1
			self.writer.write(frame)

		if nrof_faces > 0:
			if self.start_rec==False:
				print(datetime.now())
				self.save_name='{}.avi'.format(datetime.now())
			self.start_rec=True
			det = bounding_boxes[:, 0:4]
			img_size = np.asarray(frame.shape)[0:2]

			cropped = []
			scaled = []
			scaled_reshape = []
			bb = np.zeros((nrof_faces,4), dtype=np.int32)
			self.TRAIN_FRAMES.append(frame)
			for i in range(nrof_faces):
				emb_array = np.zeros((1, self.embedding_size))

				bb[i][0] = det[i][0]
				bb[i][1] = det[i][1]
				bb[i][2] = det[i][2]
				bb[i][3] = det[i][3]

				# inner exception
				if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
					print('face is too close')
					continue

				cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
				cropped[i] = facenet.flip(cropped[i], False)
				scaled.append(cv2.resize(cropped[i], (self.image_size, self.image_size)))
				scaled[i] = cv2.resize(scaled[i], (self.input_image_size,self.input_image_size),interpolation=cv2.INTER_CUBIC)
				scaled[i] = facenet.prewhiten(scaled[i])
				scaled_reshape.append(scaled[i].reshape(-1,self.input_image_size,self.input_image_size,3))
				feed_dict = {self.images_placeholder: scaled_reshape[i], self.phase_train_placeholder: False}
				emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
				predictions = self.model.predict_proba(emb_array)
				# print(predictions)
				if np.max(predictions) > 0.90:
					self.prediction_class[np.argmax(predictions)+1]+=1
				else:
					self.prediction_class[0]+=1
				self.num_imgs+=1


	def get_output(self):
		if self.num_imgs>0:
			return self.classes[np.argmax(self.prediction_class/self.num_imgs)]

	def reset(self):
		self.prediction_class=np.zeros((len(self.classes)+1))
		self.num_imgs=0
		self.start_rec=False
		self.save_name='out.avi'
		self.TRAIN_FRAMES=[]

	def save(self,label,to_save=True):
		if to_save:
			dir_=os.path.join(os.getcwd(),'face','my_classes',label)
			if not os.path.exists(dir_):
				os.makedirs(dir_)
			shutil.copyfile(os.path.join(os.getcwd(),'face',self.save_name),os.path.join(dir_,self.save_name))
		os.remove(os.path.join(os.getcwd(),'face',self.save_name))

	def re_train(self):
		print('MAKING FACES')
		img_to_face(self.MY_IMG_DIRECTORY,self.MY_FACE_DIRECTORY)
		print('FACES DONE')

		print('START RE-TRAINING')
		train_again(self.MY_FACE_DIRECTORY)
		print('RE-TRAINING DONE')

	def remove_class(self,labels):
		for label in labels:
			shutil.rmtree(os.path.join(self.MY_IMG_DIRECTORY,label))
			shutil.rmtree(os.path.join(self.MY_FACE_DIRECTORY,label))
		self.re_train()

	def add_class(self,label):
		PATH=os.path.join(self.MY_IMG_DIRECTORY,label)
		if not os.path.exists(PATH):
			os.makedirs(PATH)

		for frame in self.TRAIN_FRAMES:
			cv2.imwrite(os.path.join(PATH,'ActiOn_{}.png'.format(len[os.listdir(PATH)])),frame)
		print('IMAGES ADDED IN DIRECTORY')
		self.re_train()


if __name__=='__main__':
	fm=face_model()
	delete_state=input('Do you want to delete a label  Y/n   \t--> ')
	while delete_state=='Y' or delete_state=='y':
		delete_label=input(' Enter label : possible inputs -- {}  \t press any other to skip'.format(fm.classes))
		if delete_label not in fm.classes:
			print('not a valid label ')
			delete_state=input('Do you want to delete a label  Y/n   \t--> ')
		else:
			fm.remove_class([delete_label])

	video_capture=cv2.VideoCapture(0)
	ret, frame = video_capture.read()
	while True:
		ret,frame=video_capture.read()
		if ret:	
			cv2.imshow('ana',frame)
			fm.predict(frame)
			k = cv2.waitKey(30) & 0xff
			if k==27:
				break
	video_capture.release()
	cv2.destroyAllWindows()

	pred=fm.get_output()
	if pred!=None and pred!='Unknown':
		print(pred)
		fm.save(pred)
	# fm.remove_class(['Aditya Raj'])
	if pred=='Unknown':
		print('Label -->  Unknown')
		access=input('allow Unknown User Access Y/n   \t -->')
		if access=='Y' or access=='y':
			new_username=input('give new username\t--> ')
			fm.add_class(new_username)
			fm.save(new_username)
		else:
			print('USER ACCESS DENIED')
	fm.reset()

	test_yn=input(' Wanna Test Y/N   \t --> ')
	if test_yn=='Y' or test_yn=='y':
		test()



