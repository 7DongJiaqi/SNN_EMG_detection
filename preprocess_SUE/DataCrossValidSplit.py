import glob
import os.path
import numpy
from shutil import copyfile
from EMG_training import featureExtraction
import numpy as np
import scipy.signal as signal

def split(datapath):

    split_path = 'EMG_Data_'+datapath.split('/')[-3]+'_'+datapath.split('/')[-1]
    if not os.path.exists(split_path):
        os.mkdir(split_path)

    test_ratio = 0.2
    Folds = 10

    fileList  = sorted(glob.glob(os.path.join(datapath, "*.csv")))
    numOf_test_samples = test_ratio*round(len(fileList))

    for fold in range(Folds):
        fold_path = ('/').join((split_path,'fold'+str(fold)))
        train_path = ('/').join((fold_path,'train'))
        test_path = ('/').join((fold_path,'test'))

        if not os.path.exists(fold_path):
            os.mkdir(fold_path)
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        test_samples =  numpy.random.permutation(len(fileList))[:int(numOf_test_samples)]


        for idx,file in enumerate(fileList):
            if idx in test_samples:
                copy_path = ('/').join((test_path, file.split('/')[-1]))
                copyfile(file, copy_path)
            else:
                copy_path = ('/').join((train_path,file.split('/')[-1]))
                copyfile(file, copy_path)



def MidTermSplit(source,datapath,outFolderName):

    #create folder to save mid-term feature vectors
    if not os.path.exists(outFolderName):
        os.mkdir(outFolderName)
    if not os.path.exists('../original_labels_'+source+'/'):
        os.mkdir('../original_labels_'+source+'/')
    if not os.path.exists('../original_times_'+source+'/'):
            os.mkdir('../original_times_'+source+'/')
    print(datapath)
    fileList = sorted(glob.glob(os.path.join(datapath, "*.csv")))

    # data = {}
    # data['user'] = {}
    user = []
    exercise = []
    repetition = []
    time = []
    emg_raw = []
    gt_labels = []

    print (fileList)
    for file in fileList:

        with open(file, 'r') as f:
            x = f.readlines()
            if not x:
                continue
            time.append([float(label.split(',')[0]) for label in x])
            emg_raw.append([float(label.split(',')[1]) for label in x])
            gt_labels.append([int(label.split(',')[2].rstrip()) for label in x])
        f.close

        emg_raw[-1] = list(signal.medfilt(emg_raw[-1], 11))


        # split the sample into the positive and negative classes
        ###
        feature_vectors, labels = featureExtraction(emg_raw[-1], time[-1], gt_labels[-1],2,1,0.25,0.25)
        feature_vectors_nofatigue = []
        feature_vectors_fatigue = []
        #sys.exit()
        for i, w in enumerate(labels):
            if w == 0:
                feature_vectors_nofatigue.append(feature_vectors[:, i])
            else:
                feature_vectors_fatigue.append(feature_vectors[:, i])
        if source == 'Study1':
            user = file.split('\\')[-1].split('E')[0][1:]
            exercise = file.split('\\')[-1].split('R')[0][-1]
            repetition = file.split('\\')[-1].split('.')[0][-1]
        elif source == 'Study2.1' or source == 'Study2.2':
            user = file.split('\\')[-1].split('E')[0][1:]
            exercise = file.split('\\')[-1].split('.')[0][-1]
            repetition = '1'
        print("outFolderName",outFolderName,"user",user,"exercise",exercise,"","repetition",repetition)
        np.savez(outFolderName+'/'+('_').join((user,exercise,repetition,'NF')),np.asarray(feature_vectors_nofatigue))
        np.savez(outFolderName+'/'+('_').join((user,exercise,repetition,'F')),np.asarray(feature_vectors_fatigue))
        gt_labels[-1][-1] = 1
        np.savez('../original_labels_'+source+'/'+('_').join((user,exercise,repetition)),np.asarray(gt_labels[-1]))
        np.savez('../original_times_'+source+'/'+('_').join((user,exercise,repetition)),np.asarray(time[-1]))



if __name__ == '__main__':
    MidTermSplit('Study1','C:/study/EMG/Fatigue_Data/Study2/EMG', 'C:/study/EMG/data/Study1_medfilt11_EMG')