import pandas as pd 
import os

class save_and_display:
	def __init__(self):
		self.logs_all=None
		self.CLASSES=None

	def create_csv(self):
		PATH_GAIT_LOGS=os.path.join(os.getcwd(),'gait','my_classes')
		PATH_FACE_LOGS=os.path.join(os.getcwd(),'face','my_classes')

		assert os.listdir(PATH_FACE_LOGS)==os.listdir(PATH_GAIT_LOGS)

		self.CLASSES=os.listdir(PATH_FACE_LOGS) 


		log_class_name=[]
		log_video_addr=[]
		log_time=[]
		for class_ in self.CLASSES:
			for y in os.listdir(os.path.join(PATH_FACE_LOGS,class_)):
				log_class_name.append(class_)
				log_video_addr.append(os.path.join(PATH_FACE_LOGS,class_,y))
				log_time.append(y[:-4])  # convert to datetime


		logs_face=pd.DataFrame()
		logs_face['Class_name']=log_class_name
		logs_face['Video_addr']=log_video_addr
		logs_face['Time']=log_time
		logs_face=logs_face.sort_values(['Class_name', 'Time'], ascending=[True, True])


		log_class_name=[]
		log_video_addr=[]
		log_time=[]
		for class_ in self.CLASSES:
			for y in os.listdir(os.path.join(PATH_GAIT_LOGS,class_)):
				if y=="human":
					continue
				log_class_name.append(class_)
				log_video_addr.append(os.path.join(PATH_GAIT_LOGS,class_,y))
				log_time.append(y[:-4])  # convert to datetime

		logs_gait=pd.DataFrame()
		logs_gait['Class_name']=log_class_name
		logs_gait['Video_addr']=log_video_addr
		logs_gait['Time']=log_time
		logs_gait=logs_gait.sort_values(['Class_name', 'Time'], ascending=[True, True])


		self.logs_all=pd.DataFrame()
		self.logs_all['Class_name']=logs_face['Class_name']
		self.logs_all['Video_addr_face']=logs_face['Video_addr']
		self.logs_all['Video_addr_gait']=logs_gait['Video_addr']
		self.logs_all['Time_face']=logs_face['Time']
		self.logs_all['Time_gait']=logs_gait['Time']


	def get_logs_by_time(self,start=None,end=None,csv_file=None):

		try:
			if csv_file==None:
				csv_file=self.logs_all
		except:
			start=start
		print(start,end)

		if start==None and end==None:
			time_log=csv_file
		elif start==None:
			time_log=csv_file[csv_file['Time_face']<end]
		elif end==None:
			time_log=csv_file[csv_file['Time_face']>start]
		else:
			time_log=csv_file[csv_file['Time_face']>start][csv_file['Time_face']<end]

		print(time_log)

		return time_log

	def get_logs_by_name(self,name=[],csv_file=None):

		print(csv_file)

		try:
			if csv_file==None:
				csv_file=self.logs_all
		except:
			name=name
		name_log=pd.DataFrame(columns=csv_file.columns)

		if len(name)==0:
			return csv_file.sort_values(['Time_face'],ascending=[True])
			
		for x in name:
			name_log_=csv_file[csv_file['Class_name']==x]
			name_log=pd.concat([name_log,name_log_])
		return name_log.sort_values(['Time_face'],ascending=[True])

if __name__=='__main__':
	svd=save_and_display()
	svd.create_csv()
	x={'year': ['2020', '2020'],'month': ['1', '3'],'day': ['31', '8'],'hour':['2','5'],'minute':['21','54'],'second':['2','5']}
	start=x['year'][0]+'-'+x['month'][0]+'-'+x['day'][0]+' '+x['hour'][0]+':'+x['minute'][0]+':'+x['second'][0]
	end=x['year'][1]+'-'+x['month'][1]+'-'+x['day'][1]+' '+x['hour'][1]+':'+x['minute'][1]+':'+x['second'][1]

	print(start,end)
	time_log=svd.get_logs_by_time(start,end)
	print("TIME_LOG ")
	print(time_log)
	print('*'*100)

	name_log=svd.get_logs_by_name(name=['dhruv','ishan'])
	print("NAME_LOG ")
	print(name_log)
	print('*'*100)