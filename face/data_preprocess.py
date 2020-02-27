from preprocess import preprocesses
import os

def img_to_face(input_datadir=os.path.join(os.getcwd(),'training_files','img'),output_datadir=os.path.join(os.getcwd(),'training_files','face')):

	obj=preprocesses(input_datadir,output_datadir)
	nrof_images_total,nrof_successfully_aligned=obj.collect_data()

	print('Total number of images: %d' % nrof_images_total)
	print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

if __name__=='__main__':
	img_to_face()