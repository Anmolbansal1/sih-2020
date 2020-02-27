from flask import Flask, render_template
from flask import jsonify
from flask import request
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
from flask_sqlalchemy import SQLAlchemy
import json
import sys
import os

app=Flask(__name__,template_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///records.sqlite3'

sys.path.append(os.path.join(os.getcwd(),'gait'))

from use_this import run_this



inst=run_this()

def decd(b64_string):
	# reconstruct image as an numpy array
	image=b64_string[22:]
	
	image = Image.open(BytesIO(base64.b64decode(image)))
	img = np.array(image)
	img=img[:,:,:3]
	img=cv2.resize(img,(299,299))
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



@app.route('/',methods=["GET"])
def home():
	return render_template('index.html')

@app.route('/gait', methods=['POST'])
def record():
	# get images and decode them
	# print(request.form)
	data = request.form.getlist("x[]")
	# imgs = request.form["imgs"]
	proccesed = []
	
	for img in data:
		proccesed.append(decd(img))

	# then pass to model

	label,probs = inst.predict(np.asarray(proccesed))

	# record = records(json.dumps(probs))
	# db.session.add(record)
	# db.session.commit()

	print(probs)

	return {'key': label}


if __name__ == '__main__':
	db.create_all()
	app.run(debug=True)
