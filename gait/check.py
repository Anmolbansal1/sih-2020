import os
import numpy as np
import cv2
from human_pose_nn import HumanPoseIRNetwork
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread


net_pose = HumanPoseIRNetwork()
net_pose.restore('./gait/models/Human3.6m.ckpt')

def read_video(video_addr):
    imgs=[]
    video_capture = cv2.VideoCapture(video_addr)
    while True:
        ret, frame = video_capture.read()
        print(frame)
        try:
            frame=np.reshape(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(299,299)),(1,299,299,3))
        except:
            break
        imgs.append(frame)
        break
    print(len(imgs))
    return imgs

imgs=read_video('./gait/training_files/videos/Bansal/002.mp4')
for img in imgs:
    print(img)
    if len(np.shape(img))==3:
        img_batch = np.expand_dims(img, 0)
    else:
        img_batch=img
    y, x, a = net_pose.estimate_joints(img_batch)
    y, x, a = np.squeeze(y), np.squeeze(x), np.squeeze(a)

    joint_names = [
        'right ankle ',
        'right knee ',
        'right hip',
        'left hip',
        'left knee',
        'left ankle',
        'pelvis',
        'thorax',
        'upper neck',
        'head top',
        'right wrist',
        'right elbow',
        'right shoulder',
        'left shoulder',
        'left elbow',
        'left wrist'
    ]

    # Print probabilities of each estimation
    for i in range(16):
        print('%s: %.02f%%' % (joint_names[i], a[i] * 100))

    # Create image
    colors = ['r', 'r', 'b', 'm', 'm', 'y', 'g', 'g', 'b', 'c', 'r', 'r', 'b', 'm', 'm', 'c']
    for i in range(16):
        if i < 15 and i not in {5, 9}:
            plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color = colors[i], linewidth = 5)

    plt.imshow(np.squeeze(img))
    plt.show()