from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork
import os
import numpy as np 
import pickle
from datetime import datetime
import shutil
import cv2
from run import everything 

class run_this:
	def __init__(self):
		self.net_pose=HumanPoseIRNetwork()
		self.net_gait=GaitNetwork()
		self.net_pose.restore(os.path.join(os.getcwd(),'gait','models','Human3.6m.ckpt'))
		self.net_gait.restore(os.path.join(os.getcwd(),'gait','models','H3.6m-GRU-1.ckpt'))

		self.label_filename=os.path.join(os.getcwd(),'gait','saved_files','labelencoder.pkl')
		self.classifier_filename=os.path.join(os.getcwd(),'gait','saved_files','classifier.pkl')

		self.label_filename_vis=os.path.join(os.getcwd(),'gait','saved_files_vis','labelencoder.pkl')
		self.classifier_filename_vis=os.path.join(os.getcwd(),'gait','saved_files_vis','classifier.pkl')

		self.label_filename_temp=os.path.join(os.getcwd(),'gait','saved_files_temp','labelencoder.pkl')
		self.classifier_filename_temp=os.path.join(os.getcwd(),'gait','saved_files_temp','classifier.pkl')

		self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

		#### UNCOMMENT 
		self.everything_=everything(net_pose=self.net_pose,net_gait=self.net_gait)

	def predict(self,frames): # pose_location (input)

		# frames=process_imgs(frames,pose_location)

		features=self.net_pose.feed_forward_features(frames)
		features,_=self.net_gait.feed_forward(features)

		features=np.reshape(features,[1,-1])
		with open(self.label_filename, 'rb') as outfile:
			le=pickle.load(outfile)

		with open(self.classifier_filename, 'rb') as outfile:
			model=pickle.load(outfile)

		with open(self.label_filename_temp, 'rb') as outfile:
			le_temp=pickle.load(outfile)

		with open(self.classifier_filename_temp, 'rb') as outfile:
			model_temp=pickle.load(outfile)

		with open(self.label_filename_vis, 'rb') as outfile:
			le_vis=pickle.load(outfile)

		with open(self.classifier_filename_vis, 'rb') as outfile:
			model_vis=pickle.load(outfile)

		# print(model.classes_)
		output,probs=model.predict(features),model.predict_proba(features)
		label=le.inverse_transform(output)

		output_temp,probs_temp=model_temp.predict(features),model_temp.predict_proba(features)
		label_temp=le_temp.inverse_transform(output_temp)

		output_vis,probs_vis=model_vis.predict(features),model_vis.predict_proba(features)
		label_vis=le_vis.inverse_transform(output_vis)

		if np.max(probs)>np.max(probs_vis) and np.max(probs)>np.max(probs_temp):
			return label[0],probs,le
		elif np.max(probs_temp)>np.max(probs_vis)
			return label_temp[0],probs_temp,le_temp
		else
			return label_vis[0],probs_vis,le_vis



	def make_video(self,frames,save_name):
		(h, w) = frames[0].shape[:2]
		writer = cv2.VideoWriter(save_name, self.fourcc, 10,(w, h), True)
		for frame in frames:
			(h, w) = frame.shape[:2]
			frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			writer.write(frame)

	def save(self,frames,label,frame_human=None,type_='PERM'):
		
		if type_=='PERM':
			dir_=os.path.join(os.getcwd(),'gait','my_classes',label)
		if type_=='VIS':
			dir_=os.path.join(os.getcwd(),'gait','my_classes_vis',label)
		if type_=='TEMP':
			dir_=os.path.join(os.getcwd(),'gait','my_classes_temp',label)

		if not os.path.exists(dir_):
			os.makedirs(dir_)

		file_name='{}.mp4'.format(datetime.now())
		save_name=os.path.join(dir_,file_name)
		self.make_video(frames,save_name)


		if frame_human==None:
			return

		
		dir_=os.path.join(dir_,'human')
		if not os.path.exists(dir_):
			os.makedirs(dir_)
		file_name='{}.mp4'.format(datetime.now())
		save_name=os.path.join(dir_,file_name)
		self.make_video(frames,save_name)


	def delete(self,label,type_):
		if type_=='PERM':
			path_video,path_frames=os.path.join(os.getcwd(),'gait','training_files','videos',label),os.path.join(os.getcwd(),'gait','training_files','frames',label)
			dir_=os.path.join(os.getcwd(),'gait','my_classes',label)
		if type_=='VIS':
			path_video,path_frames=os.path.join(os.getcwd(),'gait','training_files_vis','videos',label),os.path.join(os.getcwd(),'gait','training_files_vis','frames',label)
			dir_=os.path.join(os.getcwd(),'gait','my_classes_vis',label)
		if type_=='TEMP':
			path_video,path_frames=os.path.join(os.getcwd(),'gait','training_files_temp','videos',label),os.path.join(os.getcwd(),'gait','training_files_temp','frames',label)
			dir_=os.path.join(os.getcwd(),'gait','my_classes_temp',label)

		shutil.rmtree(path_frames)
		shutil.rmtree(dir_)
		shutil.rmtree(path_video)
		self.train(type_)

	def add_frames(self,frames,label,type_):
		if type_=='PERM':
			PATH=os.path.join(os.getcwd(),'gait','training_files','videos',label)
		if type_=='VIS':
			PATH=os.path.join(os.getcwd(),'gait','training_files_vis','videos',label)
		if type_=='TEMP':
			PATH=os.path.join(os.getcwd(),'gait','training_files_temp','videos',label)

		file_name='{}.mp4'.format(len(os.listdir(PATH)))
		save_name=os.path.join(PATH,file_name)
		self.make_video(frames,save_name)
	
	def add_user(self,frames,label,type_):
		if type_=='PERM':
			PATH=os.path.join(os.getcwd(),'gait','training_files','videos',label)
		if type_=='VIS':
			PATH=os.path.join(os.getcwd(),'gait','training_files_vis','videos',label)
		if type_=='TEMP':
			PATH=os.path.join(os.getcwd(),'gait','training_files_temp','videos',label)

		if not os.path.exists(PATH):
			os.makedirs(PATH)
			print("ADDED NEW USER")

		file_name='{}.mp4'.format(len(os.listdir(PATH)))
		save_name=os.path.join(PATH,file_name)
		self.make_video(frames,save_name)
		self.train(type_)


	# def train_from_archive(self):

	## VERIFY PATHS ####

	# 	for dir_ in os.listdir(self.video_log_directory):
	# 		list_of_training_videos_path=keep_latest_list(os.path.join(self.video_log_directory,dir_))

	# 	for x in list_of_training_videos_path:
	# 		shutil.copyfile(x,self.train_folder)

	# 	self.train()
	
	def train(self):
		print("TRAINING STARTED")
		self.everything_.train()


if __name__=='__main__':
	a=run_this()
	# a.everything_.model.classes_()
	a.train()




