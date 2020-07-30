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
import time

def no0(x):
	return (max(0,x))
	
def test():
	modeldir = os.path.join(os.getcwd(),'face','model','20180402-114759.pb')
	classifier_filename = os.path.join(os.getcwd(),'face','class','classifier.pkl')
	npy=os.path.join(os.getcwd(),'face','npy')
	train_img=os.path.join(os.getcwd(),'face','training_files','img')
	label_filename=os.path.join(os.getcwd(),'face','class','label_dict.pkl')
	with open(label_filename, 'rb') as outfile:
		label_dict=pickle.load(outfile)
	classes=[class__ for _,class__ in label_dict.items()]
	prev=time.time()
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

			minsize = 20  # minimum size of face
			threshold = [0.6, 0.7, 0.7]  # three steps's threshold
			factor = 0.709  # scale factor
			image_size = 182
			input_image_size = 160
			
			HumanNames = os.listdir(train_img)
			HumanNames.sort()

			print('Loading feature extraction model')
			facenet.load_model(modeldir)

			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			print(tf.shape(images_placeholder))
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			embedding_size = embeddings.get_shape()[1]

			classifier_filename_exp = os.path.expanduser(classifier_filename)
			with open(classifier_filename_exp, 'rb') as infile:
				(model, class_names) = pickle.load(infile)

			print('Start Recognition!')
			video_capture=cv2.VideoCapture(0)
			while True:
				ret, frame = video_capture.read()
				if ret == False:
					break
				frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

				if frame.ndim == 2:
					frame = facenet.to_rgb(frame)
				print(frame.shape)
				frame = frame[:, :, 0:3]
				bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
				nrof_faces = bounding_boxes.shape[0]
				print('Face Detected: %d' % nrof_faces)

				if nrof_faces > 0:
					det = bounding_boxes[:, 0:4]
					img_size = np.asarray(frame.shape)[0:2]

					cropped = []
					scaled = []
					scaled_reshape = []
					bb = np.zeros((nrof_faces,4), dtype=np.int32)

					for i in range(nrof_faces):
						emb_array = np.zeros((1, embedding_size))

						bb[i][0] = det[i][0]
						bb[i][1] = det[i][1]
						bb[i][2] = det[i][2]
						bb[i][3] = det[i][3]

						# inner exception
						# if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
						#     print('face is too close')
						#     continue

						bb[i][0] = no0(bb[i][0])
						bb[i][1] = no0(bb[i][1])
						bb[i][2] = no0(bb[i][2])
						bb[i][3] = no0(bb[i][3])

						cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
						cropped[i] = facenet.flip(cropped[i], False)
						scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
						scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
											   interpolation=cv2.INTER_CUBIC)
						scaled[i] = facenet.prewhiten(scaled[i])
						scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
						
						feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
						emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
						predictions = model.predict_proba(emb_array)
						text_x = bb[i][0]
						text_y = bb[i][3] + 20

						cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
						print(predictions)
						if np.max(predictions) > 0.50:
							print(np.max(predictions))
							cv2.putText(frame, classes[np.argmax(predictions)], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
											1, (0, 0, 255), thickness=1, lineType=2)
						else:
							cv2.putText(frame, 'UNKNOWN', (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
											1, (0, 0, 255), thickness=1, lineType=2)

				cv2.imshow('Image', frame)
				k = cv2.waitKey(30) & 0xff
				if k==27:
				   break

				fps=1/(time.time()-prev)
				print('FPS :: ',fps)
				prev=time.time()

			video_capture.release()
			cv2.destroyAllWindows()


if __name__=='__main__':
	test()