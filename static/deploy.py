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

def decd(b64_string):
    # reconstruct image as an numpy array
    image=b64_string[22:]
    image = Image.open(BytesIO(base64.b64decode(b64_string)))
    img = np.array(image)
    return img

def encd(img):
    # converts numpy array as base 64 encoded image
    pil_img = Image.fromarray(img.astype('uint8'))
    buff = BytesIO()
    pil_img.save(buff, format="png")
    image = base64.b64encode(buff.getvalue()).decode("utf-8")
    return image

app=Flask(__name__,template_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///records.sqlite3'

db = SQLAlchemy(app)
class records(db.Model):
    id = db.Column('id', db.Integer, primary_key = True)
    probs = db.Column(db.String(100))

    def __init__(self, probs):
        self.probs = probs



@app.route('/',methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    # get images and decode them
    imgs = request.imgs
    proccesed = []
    for img in imgs:
        proccesed.append(decode(img))

    # then pass to model
    probs = model(proccesed)

    record = records(json.dumps(probs))
    db.session.add(record)
    db.session.commit()

    print(record)

    return {'key': 'urdu_faarsi'}


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
