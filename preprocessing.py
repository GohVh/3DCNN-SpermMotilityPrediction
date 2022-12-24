import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
import shutil

def d1gen_frame(videofolder, outputfolder, stride_num, extract_num, d1_imgsize):
	if not os.path.exists(outputfolder):
		os.mkdir(outputfolder)

	video_list = os.listdir(videofolder)

	for video in video_list:
		video_file = os.path.join(videofolder, video)
		output_path_org = os.path.join(outputfolder, video)

		if not os.path.exists(output_path_org):
			os.mkdir(output_path_org)

		cap = cv2.VideoCapture(video_file)
		length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		start_point = 1

		print(video_file)
		print("length:", length)
		print("Skip stride=", stride_num)

		while cap.isOpened():
			for i in tqdm(range(extract_num)):
				outfile_org = os.path.join(output_path_org, str(i) + ".png")

				cap.set(1, start_point)
				ret, frame1 = cap.read()

				cap.set(1, start_point + stride_num)
				ret, frame2 = cap.read()

				hsv = np.zeros_like(frame1)
				hsv[...,1] = 255
				grayframe1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
				grayframe2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
				flow = cv2.calcOpticalFlowFarneback(grayframe1, grayframe2,
							None,
							0.5, 3, 15, 3, 5, 1.2, 0)
				magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
				hsv[..., 0] = angle * 180 / np.pi / 2
				hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
				rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

				rgb_resized = cv2.resize(rgb, (d1_imgsize, d1_imgsize))

				cv2.imwrite(outfile_org, rgb_resized)

				k = cv2.waitKey(1) & 0xff

				if k == 27:
					break

				start_point += stride_num

			cap.release()
			cv2.destroyAllWindows()

def rename_dir(extractfolder, outputfolder):
	folder_list = os.listdir(extractfolder)
	
	if not os.path.exists(outputfolder):
		os.mkdir(outputfolder)
		
	for imagefolder in folder_list:
		sample_num = int((imagefolder.split('_'))[0])
		newimagefolder = os.path.join(extractfolder, str(sample_num))
		os.rename(os.path.join(extractfolder, imagefolder), newimagefolder)
		image_list = os.listdir(newimagefolder)
		
		for image in image_list:
			image_num = int((image.split('.'))[0])
			oldimgadd = os.path.join(newimagefolder, image)
			renamedimg = f'{sample_num}_{image_num}.png'
			newimgadd = os.path.join(newimagefolder, renamedimg)
			os.rename(oldimgadd, newimgadd)
			shutil.copy(newimgadd, outputfolder)
			
def stack(extractfolder, stackedfolder, stackframe_num, batchnum, h,w):
	
	if not os.path.exists(stackedfolder):
		os.mkdir(stackedfolder)
		
	folder_list = os.listdir(extractfolder)
	videoid_list = []
	
	for imagefolder in folder_list:
		videoid = int((imagefolder.split('_'))[0])
		videoid_list.append(videoid)
		os.rename(os.path.join(extractfolder, imagefolder), os.path.join(extractfolder, str(videoid)))
		
	videoid_list = sorted(videoid_list)
	
	for videoid in videoid_list:
		img_arr = np.empty([3, h, w, stackframe_num])
		image_num=0

		for batchid in range(batchnum):
			for i in range(stackframe_num):
				img_add = f'{extractfolder}/{videoid}/{image_num}.png'
				temp_img = cv2.imread(img_add)
				
				for rgbnum in range(3):
					img_arr[rgbnum,:,:,i] = temp_img[:,:,rgbnum]
					
				image_num+=1
				
			np.save(f'{stackedfolder}/{videoid}_{batchid}.npy', img_arr)

def shuffle_pointer(maindir, filename):
	pointerdf = pd.read_csv(f'{maindir}{filename}.csv')
	pointerdf_shuffled = shuffle(pointerdf)
	pointerdf_shuffled.to_csv(f'{maindir}{filename}_shuffled.csv',index=False)

def split_kfold(maindir, datasetdir, filename, kfoldnum, scalenum):
	pointerdf = pd.read_csv(f'{maindir}{filename}_shuffled.csv')
	totalsample = pointerdf.shape[0]
	quarter = totalsample//kfoldnum
	remain = totalsample%kfoldnum
	d = {}
	
	for i in range(kfoldnum):
		if i != (kfoldnum-1):
			d[f'fold{i+1}'] = pointerdf.iloc[i*quarter:(i+1)*quarter,:]
		else:
			d[f'fold{i+1}'] = pointerdf.iloc[i*quarter:((i+1)*quarter)+remain,:]
			
	for i in range(kfoldnum):
		tempd = d.copy()
		testdf = pd.DataFrame()
		traindf = pd.DataFrame()
		
		testdf = tempd[f'fold{i+1}']
		tempd.pop(f'fold{i+1}')
		for x in tempd:
			traindf = traindf.append(tempd[x])
		
		testdf = shuffle(add_dir(scale(testdf, scalenum), datasetdir))
		traindf = shuffle(add_dir(scale(traindf, scalenum), datasetdir))
		testdf.to_csv(f'{maindir}testf{i+1}.csv',index=False)
		traindf.to_csv(f'{maindir}trainf{i+1}.csv',index=False)
		
def scale(df, scalenum):
	pointerdf = df
	temp_batchid = pd.DataFrame(columns=['BATCHID'])
	pointer_duplicated = pointerdf.apply(lambda row: row.repeat(scalenum), axis=0).reset_index().drop(columns=['index'])
	for i in range(scalenum):
		temp_batchid = temp_batchid.append({'BATCHID': i}, ignore_index=True)
	
	temp_batchid = temp_batchid.append([temp_batchid]*(pointerdf.shape[0]-1), ignore_index=True)
	pointerdf_updated = pd.concat([pointer_duplicated, temp_batchid], axis=1).astype(int)
	pointerdf_updated = pointerdf_updated[['VIDEOID','BATCHID','PR','NPR','IM','CONC']]
	return pointerdf_updated

def add_dir(df, datasetdir):
	data1add = f'{datasetdir}data1'
	data2add = f'{datasetdir}data2'
	pointerdf = df
	tempdf1 = pd.DataFrame(columns=['ORI_DIR1'])
	tempdf2 = pd.DataFrame(columns=['ORI_DIR2'])

	for i in range(pointerdf.shape[0]):
		videoid = str(pointerdf.iloc[i,0])
		batchid = str(pointerdf.iloc[i,1])

		dir_add1 = f'{data1add}/{videoid}_{batchid}.npy'
		dir_add2 = f'{data2add}/{videoid}_{batchid}.png'

		tempdf1 = tempdf1.append({'ORI_DIR1': dir_add1}, ignore_index=True)
		tempdf2 = tempdf2.append({'ORI_DIR2': dir_add2}, ignore_index=True)

	pointerdf_updated = pd.concat([pointerdf, tempdf1], axis=1)
	pointerdf_updated = pd.concat([pointerdf_updated, tempdf2], axis=1)
	return pointerdf_updated
