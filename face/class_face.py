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

def no0(x):
	return (max(0,x))

class face_model:
	def __init__(self):
		self.modeldir = os.path.join(os.getcwd(),'face','model','20180402-114759.pb')
		self.classifier_filename = os.path.join(os.getcwd(),'face','class','classifier.pkl')
		self.npy=os.path.join(os.getcwd(),'face','npy')
		self.MY_FACE_DIRECTORY=os.path.join(os.getcwd(),'face','training_files','face')
		self.MY_IMG_DIRECTORY=os.path.join(os.getcwd(),'face','training_files','img')
		self.MY_FACE_DIRECTORY_TEMP=os.path.join(os.getcwd(),'face','training_files_temp','face')
		self.MY_IMG_DIRECTORY_TEMP=os.path.join(os.getcwd(),'face','training_files_temp','img')
		self.MY_FACE_DIRECTORY_VIS=os.path.join(os.getcwd(),'face','training_files_vis','face')
		self.MY_IMG_DIRECTORY_VIS=os.path.join(os.getwd(),'face','training_files_vis','img')
		self.TRAIN_FRAMES=[]
		self.write_frames=[]
		### DEBUGGER

		# constructor takes in list of classes
		self.save_name=None
		# self.prediction_class=np.zeros((len(self.classes)))
		self.classifier_filename_exp = os.path.expanduser(self.classifier_filename)
		with open(self.classifier_filename_exp, 'rb') as infile:
			(self.model, self.class_names,self.classes) = pickle.load(infile)
		self.prediction_probs=np.zeros((1,len(self.classes)))

		self.classifier_filename_exp_vis = os.path.expanduser(self.classifier_filename_vis)
		with open(self.classifier_filename_exp_vis, 'rb') as infile:
			(self.model_vis, self.class_names_vis,self.classes_vis) = pickle.load(infile)
		self.prediction_probs_vis=np.zeros((1,len(self.classes_vis)))

		self.classifier_filename_exp_temp = os.path.expanduser(self.classifier_filename_temp)
		with open(self.classifier_filename_exp_temp, 'rb') as infile:
			(self.model_temp, self.class_names_temp,self.classes_temp) = pickle.load(infile)
		self.prediction_probs_temp=np.zeros((1,len(self.classes_temp)))

		
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


			

			self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			print('Start Recognition!')

	def predict(self,frame):

		if self.save_name==None:
			self.save_name='{}.mp4'.format(datetime.now())
		frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

		if frame.ndim == 2:
			frame = facenet.to_rgb(frame)
		frame = frame[:, :, 0:3]
		bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
		nrof_faces = bounding_boxes.shape[0]
		# print('Face Detected: %d' % nrof_faces)

		self.write_frames.append(frame)

		if nrof_faces > 0:				
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
				# if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
				# 	print('face is too close')
				# 	continue

				bb[i][0] = no0(bb[i][0])
				bb[i][1] = no0(bb[i][1])
				bb[i][2] = no0(bb[i][2])
				bb[i][3] = no0(bb[i][3])


				cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
				cropped[i] = facenet.flip(cropped[i], False)
				scaled.append(cv2.resize(cropped[i], (self.image_size, self.image_size)))
				scaled[i] = cv2.resize(scaled[i], (self.input_image_size,self.input_image_size),interpolation=cv2.INTER_CUBIC)
				scaled[i] = facenet.prewhiten(scaled[i])
				scaled_reshape.append(scaled[i].reshape(-1,self.input_image_size,self.input_image_size,3))
				feed_dict = {self.images_placeholder: scaled_reshape[i], self.phase_train_placeholder: False}
				emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
				
				predictions = self.model.predict_proba(emb_array)
				predictions_vis=self.model_vis.predict_proba(emb_array)
				prediction_temp=self.model_temp.predict_proba(emb_array)
				# print(predictions)
				self.prediction_probs+=predictions
				self.prediction_probs_temp+=prediction_temp
				self.prediction_probs_vis+=prediction_probs_vis
				# if np.max(predictions) > 0.90:
				# 	self.prediction_class[np.argmax(predictions)+1]+=1
				# else:
				# 	self.prediction_class[0]+=1
				self.num_imgs+=1


	def get_output(self):
		if self.num_imgs>0:
			self.prediction_probs/=self.num_imgs
			self.prediction_probs_vis/=self.num_imgs
			self.prediction_probs_temp/self.num_imgs
		if self.prediction_probs>self.prediction_probs_vis and self.prediction_probs>self.prediction_probs_temp:
			return self.classes[np.argmax(self.prediction_probs)],self.prediction_probs,self.classes,self.num_imgs
		elif self.prediction_probs_vis>self.prediction_probs_temp:
			return self.classes_vis[np.argmax(self.prediction_probs_vis)],self.prediction_probs_vis,self.classes_vis,self.num_imgs
		else:
			return self.classes_temp[np.argmax(self.prediction_probs_temp)],self.prediction_probs_temp,self.classes_temp,self.num_imgs

	def save(self,label,type_):
		if type_=='PERM':
			dir_=os.path.join(os.getcwd(),'face','my_classes',label)
		if type_=='VIS':
			dir_=os.path.join(os.getcwd(),'face','my_classes_vis',label)
		if type_=='TEMP':
			dir_=os.path.join(os.getcwd(),'face','my_classes_temp',label)

		(h, w) = self.write_frames[0].shape[:2]
		if not os.path.exists(dir_):
			os.makedirs(dir_)

		writer = cv2.VideoWriter(os.path.join(dir_,self.save_name), self.fourcc, 10,(w, h), True)
		for frame in self.write_frames:
			writer.write(frame)


	def reset(self):
		# self.prediction_class=np.zeros((len(self.classes)+1))
		self.num_imgs=0
		self.save_name=None
		self.TRAIN_FRAMES=[]

		self.write_frames=[]
		### DEBUGGER

		# constructor takes in list of classes

		self.classifier_filename_exp = os.path.expanduser(self.classifier_filename)
		with open(self.classifier_filename_exp, 'rb') as infile:
			(self.model, self.class_names,self.classes) = pickle.load(infile)
		self.prediction_probs=np.zeros((1,len(self.classes)))

		self.classifier_filename_exp_vis = os.path.expanduser(self.classifier_filename_vis)
		with open(self.classifier_filename_exp_vis, 'rb') as infile:
			(self.model_vis, self.class_names_vis,self.classes_vis) = pickle.load(infile)
		self.prediction_probs_vis=np.zeros((1,len(self.classes_vis)))

		self.classifier_filename_exp_temp = os.path.expanduser(self.classifier_filename_temp)
		with open(self.classifier_filename_exp_temp, 'rb') as infile:
			(self.model_temp, self.class_names_temp,self.classes_temp) = pickle.load(infile)
		self.prediction_probs_temp=np.zeros((1,len(self.classes_temp)))
		print('Start Recognition!')

	def re_train(self,type_):
		print('MAKING FACES')
		if type_=='PERM':
			img_to_face(self.MY_IMG_DIRECTORY,self.MY_FACE_DIRECTORY)
			print('FACES DONE')
			train_again(datadir=self.MY_FACE_DIRECTORY,classifier_filename=os.path.join(os.getcwd(),'face','class','classifier.pkl'))
		if type_=='VIS'
			img_to_face(self.MY_IMG_DIRECTORY_VIS,self.MY_FACE_DIRECTORY_VIS)
			print('FACES DONE')
			train_again(datadir=self.MY_FACE_DIRECTORY_VIS,classifier_filename=os.path.join(os.getcwd(),'face','class_vis','classifier.pkl'))
		if type_=='TEMP'
			img_to_face(self.MY_IMG_DIRECTORY_TEMP,self.MY_FACE_DIRECTORY_TEMP)
			print('FACES DONE')
			train_again(datadir=self.MY_FACE_DIRECTORY_TEMP,classifier_filename=os.path.join(os.getcwd(),'face','class_temp','classifier.pkl'))
		
		print('RE-TRAINING DONE')

	def remove_class(self,labels,type_):
		if type_=='PERM':
			for label in labels:
				shutil.rmtree(os.path.join(self.MY_IMG_DIRECTORY,label))
				shutil.rmtree(os.path.join(self.MY_FACE_DIRECTORY,label))
				shutil.rmtree(os.path.join(os.getcwd(),'face','my_classes',label))
		if type_=='VIS':
			for label in labels:
				shutil.rmtree(os.path.join(self.MY_IMG_DIRECTORY_VIS,label))
				shutil.rmtree(os.path.join(self.MY_FACE_DIRECTORY_VIS,label))
				shutil.rmtree(os.path.join(os.getcwd(),'face','my_classes_vis',label))
		if type_=='TEMP':
			for label in labels:
				shutil.rmtree(os.path.join(self.MY_IMG_DIRECTORY_TEMP,label))
				shutil.rmtree(os.path.join(self.MY_FACE_DIRECTORY_TEMP,label))
				shutil.rmtree(os.path.join(os.getcwd(),'face','my_classes_temp',label))

		self.re_train(type_)


	def add_class(self,label,type_):
		if type_=='PERM':
			PATH=os.path.join(self.MY_IMG_DIRECTORY,label)
		if type_=='VIS':
			PATH=os.path.join(self.MY_IMG_DIRECTORY_VIS,label)
		if type_=='TEMP':
			PATH=os.path.join(self.MY_IMG_DIRECTORY_TEMP,label)

		if not os.path.exists(PATH):
			os.makedirs(PATH)

		for frame in self.TRAIN_FRAMES:
			cv2.imwrite(os.path.join(PATH,'ActiOn_{}.png'.format(len(os.listdir(PATH)))),frame)
		print('IMAGES ADDED IN DIRECTORY')
		self.re_train(type_)

		
	def add_frames(self,label,type_):
		if type_=='PERM':
			PATH=os.path.join(self.MY_IMG_DIRECTORY,label)
		if type_=='VIS':
			PATH=os.path.join(self.MY_IMG_DIRECTORY_VIS,label)
		if type_=='TEMP':
			PATH=os.path.join(self.MY_IMG_DIRECTORY_TEMP,label)
			
		for frame in self.TRAIN_FRAMES:
			cv2.imwrite(os.path.join(PATH,'ActiOn_{}.png'.format(len(os.listdir(PATH)))),frame)
		print('IMAGES ADDED IN DIRECTORY')


if __name__=='__main__':
	THRESHOLD=0.8
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

	pred,probs_out,_,ng=fm.get_output()
	print("probs_out ",probs_out)
	if np.max(probs_out)<THRESHOLD:
		pred='Unknown'
	print(pred,probs_out)
	if pred!=None and pred!='Unknown':
		print(pred,probs_out)
		fm.save(pred)
	# fm.remove_class(['Aditya Raj'])
	if pred=='Unknown':
		print('Label -->  Unknown')
		access=input('allow Unknown User Access Y/n   \t -->')
		if access=='Y' or access=='y':
			new_username=input('give new username\t--> ')
			# fm.add_class(new_username)
			fm.save(new_username)
		else:
			print('USER ACCESS DENIED')
	fm.reset()

	test_yn=input(' Wanna Test Y/N   \t --> ')
	if test_yn=='Y' or test_yn=='y':
		test()



