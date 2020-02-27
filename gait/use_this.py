# from run import everything
from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork
import os
import numpy as np 
import pickle
from datetime import datetime
import shutil
import cv2
# from run import everything

class run_this:
	def __init__(self):
		self.net_pose=HumanPoseIRNetwork()
		self.net_gait=GaitNetwork()
		self.net_pose.restore(os.path.join(os.getcwd(),'gait','models','Human3.6m.ckpt'))
		self.net_gait.restore(os.path.join(os.getcwd(),'gait','models','H3.6m-GRU-1.ckpt'))
		self.label_filename=os.path.join(os.getcwd(),'gait','saved_files','labelencoder.pkl')
		self.classifier_filename=os.path.join(os.getcwd(),'gait','saved_files','classifier.pkl')
		self.train_inp_=os.path.join(os.getcwd(),'gait','training_files','videos')
		self.train_out_=os.path.join(os.getcwd(),'gait','training_files','frames')
		self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		# self.everything_=everything(self.train_inp_,self.train_out_)
		self.video_log_directory=os.path.join(os.getcwd(),'gait','my_classes')
		self.train_folder=os.path.join(os.getcwd(),'gait','training_files')

	def predict(self,frames):
		features=self.net_pose.feed_forward_features(frames)
		features,_=self.net_gait.feed_forward(features)

		features=np.reshape(features,[1,-1])
		with open(self.label_filename, 'rb') as outfile:
			le=pickle.load(outfile)

		with open(self.classifier_filename, 'rb') as outfile:
			model=pickle.load(outfile)

		output,probs=model.predict(features),model.predict_proba(features)
		label=le.inverse_transform(output)

		print("label = ",label)
		return label,probs


	def make_video(self,frames,save_name):
		writer = cv2.VideoWriter(save_name, self.fourcc, 10,(w, h), True)
		for frame in frames:
			(h, w) = frame.shape[:2]
			writer.write(frame)

	def save(self,frames,label):
		if to_save:
			dir_=os.path.join(os.getcwd(),'my_classes',label)
			if not os.path.exists(dir_):
				os.makedirs(dir_)
			file_name='{}.avi'.format(datetime.now())
			save_name=os.path.join(dir_,file_name)
			self.make_video(frames,save_name)

	def process_imgs(self,x):
		pass

	def delete(self,frames,label):
		path_video,path_frames=os.path.join(os.getcwd(),'gait','training_files','videos',label),os.path.join(os.getcwd(),'gait','training_files','frames',label)
		shutil.rmtree(path_frames)
		shutil.rmtree(path_video)
		self.train()

	def add_user_or_video(self,frames,label):
		PATH=os.path.join(os.getcwd(),'gait','training_files','videos',label)
		if not os.path.exists(PATH):
			os.makedirs(PATH)
			print("ADDED NEW USER")

		file_name='{}.avi'.format(len[os.listdir(PATH)])
		save_name=os.path.join(PATH,file_name)
		self.make_video(frames,save_name)
		self.train()


	# def train_from_archive(self):

	## VERIFY PATHS ####

	# 	for dir_ in os.listdir(self.video_log_directory):
	# 		list_of_training_videos_path=keep_latest_list(os.path.join(self.video_log_directory,dir_))

	# 	for x in list_of_training_videos_path:
	# 		shutil.copyfile(x,self.train_folder)

	# 	self.train()

	# def train(self):
	# 	print("TRAINING STARTED")
	# 	self.everything_.train()


if __name__=='__main__':
	a=run_this()

