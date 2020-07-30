import numpy as np

class final_model_():
	def __init__(self):
		self.users=dict()
		self.gait_frames=None
		self.gait_frames_center=None


	def get_gait_frames(self,frames_center,frames):
		self.gait_frames=frames
		self.gait_frames_center=frames_center


	def get_gait_probs(self,gait_probs):
		self.gait_probs=np.squeeze(gait_probs)

	def get_face_probs(self,face_probs):
		self.face_probs=np.squeeze(face_probs)

	def get_face_classes(self,classes):
		self.face_classes=classes


	def get_gait_classes(self,label_enc):
		self.label_enc_gait=label_enc

	def ensemble_model(self):
		# MAKE AN OPTIMIZER FOR X1*probs_face1+X2*probs_face2+...XN-1*probs_gaitk+XN*probs_gait  ---> X(shape) output do 1 layer NN 
		# MINIMIZE THE LOSS FUNCTION for class output.
		face=dict()
		gait=dict()
		print(self.face_classes)
		for x in range(len(self.face_classes)):
			face[self.face_classes[x]]=self.face_probs[x]

		for x in range(self.gait_probs.shape[0]):
			gait[self.label_enc_gait.inverse_transform([x])[0]]=self.gait_probs[x]

		print("FACE ",face)
		print("GAIT ",gait)

		probs_final=(self.face_probs+self.gait_probs)/2
		max_final_val,max_final_name=np.max(probs_final),self.face_classes[np.argmax(probs_final)]

		print("FINAL --> VALUE--> {}  INDEX--> {}".format(max_final_val,max_final_name))
		self.name=max_final_name
		return "unknown",max_final_val,self.face_probs[np.argmax(probs_final)],self.gait_probs[np.argmax(probs_final)]




