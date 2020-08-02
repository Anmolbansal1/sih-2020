
########################################################### IMPORTS #########################################################################

from flask import Flask, render_template, Response, url_for
from flask import jsonify
from flask import request
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
from flask_sqlalchemy import SQLAlchemy
import pickle
import json
import sys
import os
import cv2
import shutil



############################################################## APP CONFIGURATIONS ######################################################################



app=Flask(__name__,template_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///records.sqlite3'

sys.path.append(os.path.join(os.getcwd(),'gait'))

from use_this import run_this

sys.path.append(os.path.join(os.getcwd(),'face'))

from class_face import face_model

from final_model_class import final_model_


############################################################### INITIALIZATIONS ##############################################################################



inst=run_this()
fm=face_model()
final_model=final_model_()
THRESHOLD_GAIT=10


################################################################ UTILITY FUNCTIONS ###################################################################################


def no0(x,y):
	return (max(0,x),max(0,y))


def decd(b64_string):
	# reconstruct image as an numpy array
	image=b64_string[22:]
	
	image = Image.open(BytesIO(base64.b64decode(image)))
	img = np.array(image)
	img=img[:,:,:3]
	return img

def encd(img):
	# converts numpy array as base 64 encoded image
	pil_img = Image.fromarray(img.astype('uint8'))
	buff = BytesIO()
	pil_img.save(buff, format="png")
	image = base64.b64encode(buff.getvalue()).decode("utf-8")
	return image



db = SQLAlchemy(app)
class records(db.Model):
	id = db.Column('id', db.Integer, primary_key = True)
	probs = db.Column(db.String(100))

	def __init__(self, probs):
		self.probs = probs



############################################################################### MAIN APPLICATION ##############################################################



@app.route('/',methods=["GET"])
def home():
	return render_template('index.html')

@app.route('/users', methods=["GET"])
def users():
	return render_template('delete.html')


@app.route('/about', methods=["GET"])
def about():
	return render_template('about.html')


############################################################################### GAIT MODEL ROUTE ####################################################################
@app.route('/gait', methods=['POST'])
def record():
	# get images and decode them
	# print(request.form)
	imgs = request.form.getlist("x[]")
	bb   = request.form.getlist("bb[]")

	if len(imgs)<THRESHOLD_GAIT:
		return "FALSE ALARM"

	box=[]
	for x in bb:
		box.append((int)((float)(x)))
		
	final_img = []
	save_img = []
	
	couter = 0

	for x in range(1,len(imgs)):
		img=decd(imgs[x])
		# if(x<2):
		# 	cv2.imshow("kh",img)
		# 	cv2.waitKey(0)
		# 	cv2.destroyAllWindows()
		print(img.shape)
		save_img.append(img)
		start=no0(box[x*4+0],box[x*4+1])
		end=no0(start[0]+box[x*4+2],start[1]+box[x*4+3])
		print(start,end)
		print("*"*100)
		img1=img[start[0]:end[0],start[1]:end[1]]
		try:
			img1=cv2.resize(img1,(299,299))
		except:
			couter+=1
			continue
		final_img.append(img1)

	
	print(couter)


	# then pass to model

	gait_label,probs_gait,label_enc_gait = inst.predict(np.asarray(final_img))
	print('&'*100)
	print("GAIT LABEL  ",gait_label)
	# record = records(json.dumps(probs))
	# db.session.add(record)
	# db.session.commit
	final_model.get_gait_frames(frames_center=final_img,frames=save_img)
	final_model.get_gait_classes(label_enc_gait)
	final_model.get_gait_probs(probs_gait)
	print(probs_gait)

	return {'gait_output': gait_label}




############################################################################## FACE MODEL ROUTE #################################################################



@app.route('/feedFace', methods=['POST'])
def feedFace():
	# get images and decode them
	# print(request.form) 
	

	data = request.form['x']
	img = decd(data)

	# then pass to model
	img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	fm.predict(img)
	state="qwerty"
	return {'face_key': state}


@app.route('/getFace', methods=['GET'])
def getFace():

	# then pass to model
	
	face_pred,face_probs,class_list,num_img = fm.get_output()
	
	final_model.get_face_probs(face_probs)
	print(face_probs)

	final_model.get_face_classes(class_list)
	return {'face_output': str(face_pred),'flag':num_img}





########################################################################### ENSEMBLING ROUTE #################################################################


@app.route('/final',methods=['GET'])
def final_model_app():
	final_pred,final_prob, face_probs, gait_probs =final_model.ensemble_model()
	return {'final_answer':str(final_pred), 'final_prob': final_prob, 'face_prob': face_probs, 'gait_prob': gait_probs}



############################################################################ ACCESS ROUTE #################################################################

@app.route('/access',methods=["POST"])
def grant():
	name=request.form["user"]
	
	if name=="terrorist":
		inst.save(frames=final_model.gait_frames,label="Unknown")
		# cv2.imshow("kk",final_model.gait_frames[0])
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		fm.save('Unknown')
	else:
		fm.add_class(name)
		# cv2.imshow("kk",final_model.gait_frames[0])
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		inst.add_user(final_model.gait_frames_center,name)
		inst.save(frames=final_model.gait_frames,label=name,frame_human=final_model.gait_frames_center)
		fm.save(name)
	fm.reset()

	classifier_filename = os.path.join(os.getcwd(),'face','class','classifier.pkl')
	classifier_filename_exp = os.path.expanduser(classifier_filename)
	with open(classifier_filename_exp, 'rb') as infile:
		(_, _,classes) = pickle.load(infile)
	final_model.get_face_classes(classes)

	with open(os.path.join(os.getcwd(),"gait","saved_files","labelencoder.pkl"),'rb') as f:
		final_model.get_gait_classes(pickle.load(f))

	return {"key":"done"}

@app.route('/access2',methods=["POST"])
def add2():
	name=request.form["user"]

	fm.add_frames(name)
	# inst.add_frames()
	inst.add_frames(final_model.gait_frames_center,name)

	return {"key":"done"}



@app.route('/train',methods=['GET'])
def train_all():
	inst.train()
	fm.re_train()

	return "training all"






##############################################    SEARCH AND DELETE ROUTES   ###########################################################################


from logger import save_and_display

sad=save_and_display()

@app.route('/search',methods=['GET'])
def search_home():
	return render_template('search.html')


@app.route('/userlist',methods=['GET'])
def users_list():
	sad.create_csv()
	return {"users":sad.CLASSES}

@app.route('/finder',methods=['POST'])
def finding():
	sad.create_csv()


	st=request.form['start']
	ed=request.form['end']
	names=request.form.getlist("list[]")

	

	

	if len(st)!=0:
		st=st[0:10]+' '+st[11:]+":00"
	else:
		st=None

	if len(ed)!=0:
		ed=ed[0:10]+' '+ed[11:]+":00"
	else:
		ed=None

	x=sad.get_logs_by_time(start=st,end=ed)
	y=sad.get_logs_by_name(name=names,csv_file=x)

	list_form=[list(x) for x in y.values]

	print("*"*80)
	print(y)
	print("*"*80)
	print(x)

	print(list_form)
	return {"file" : list_form}



@app.route('/delete',methods=['GET', 'POST'])
def remuser():

	classifier_filename = os.path.join(os.getcwd(),'face','class','classifier.pkl')
	classifier_filename_exp = os.path.expanduser(classifier_filename)
	with open(classifier_filename_exp, 'rb') as infile:
		(_, _,classes) = pickle.load(infile)
	final_model.get_face_classes(classes)

	with open(os.path.join(os.getcwd(),"gait","saved_files","labelencoder.pkl"),'rb') as f:
		final_model.get_gait_classes(pickle.load(f))
	
	if request.method == 'GET':
		return {'data': final_model.face_classes}

	name=request.form['user']
	if name not in final_model.face_classes:
		return "user not found"
	fm.remove_class([name])
	inst.delete(name)

	return "deleted"



################################################################# VIDEO FEED AND CLEAN ROUTE #########################################################

import string
import random

@app.route('/video_feed',methods=['POST'])
def feeding():     # contains 3 video name address gait1,gait2,face  --- GIVE INPUT IN THIS ORDER

	#name of temp dir
	temp_dir=os.path.join(os.getcwd(),'static','temp')

	# cleaering if already exists
	if os.path.exists(temp_dir):
		shutil.rmtree(temp_dir)

	#making the new temp directory
	os.mkdir(temp_dir)

	names=request.form.getlist('url[]')
	# list of URLs which are returned
	resListURL=[]
	print(names)
	# file name for the 3 videos saved in temp

	file_name=[''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
,''.join(random.choices(string.ascii_uppercase + string.digits, k=5))]

	# iterating through the videos and saving them in temp and adding to return URL
	for x,y in zip(names,file_name):
		shutil.copyfile(x,os.path.join(os.getcwd(),'static','temp',y))   # saving videos in temp
		resListURL.append(url_for('static',filename='temp/{}'.format(y)))  # appeending URLs

	return jsonify(urls=resListURL)


############################################################# MAIN ###############################################################################


if __name__ == '__main__':
	db.create_all()
	app.run(debug=True)
