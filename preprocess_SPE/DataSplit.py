import glob
import os.path
import numpy
from shutil import copyfile
import pandas as pd
import numpy as np
import scipy.signal as signal
import FeatureExtraction_1D as f1d


def stLabelsExtraction(labels, Fs, Win, Step):

    Win = int(Win)
    Step = int(Step)

    N = len(labels)  # total number of samples
    curPos = 0
    countFrames = 0

    stLabels = []
    while (curPos + Win - 1 < N):  # for each short-term window until the end of signal
        countFrames += 1
        x = labels[curPos:curPos + Win]  # get current window

        if x.count(0) > x.count(1):
            stLabels.append(0)
        else:
            stLabels.append(1)

        curPos = curPos + Step  # update window position

    return stLabels


def mtLabelExtraction(labels, Fs, mtWin, mtStep, stWin, stStep):
    mtWinRatio = int(round(mtWin / stStep))
    mtStepRatio = int(round(mtStep / stStep))

    mtLabels = []

    stLabels = stLabelsExtraction(labels, Fs, stWin, stStep)

    curPos = 0
    N = len(stLabels)
    while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N:
                N2 = N
            curStLabels = stLabels[N1:N2]
            if curStLabels.count(0) > curStLabels.count(1):
                mtLabels.append(0)
            else:
                mtLabels.append(1)
            curPos += mtStepRatio

    return mtLabels, stLabels

def read_label(file_path):
    data = pd.read_excel(file_path, sheet_name='Sheet1')
    sensor_data = data.iloc[2:11, 1:9]

    processed_data = pd.DataFrame(index=sensor_data.index, columns=sensor_data.columns)
    for index, row in sensor_data.iterrows():
        processed_row = []
        for value in row:
            if value == 3:
                processed_row.append(1)
            elif value in [0, 1, 2]:
                processed_row.append(0)
            else:
                processed_row.append(np.nan)
        processed_data.loc[index] = processed_row
    return processed_data

def featureExtraction(raw_data, time, mW, mS, sW, sS):
    # emg_features_vectors = []
    # duration = float(time[-1] - time[0])
    duration = 240
    Fs = round(len(raw_data) / duration)


    mtWin = mW
    mtStep = mS
    stWin = sW
    stStep = sS

    [MidTermFeatures, stFeatures] = f1d.mtFeatureExtraction(raw_data, Fs, round(mtWin * Fs), round(mtStep * Fs),
                                                            round(Fs * stWin), round(Fs * stStep))




    print("MidTermFeatures",MidTermFeatures.shape[0],MidTermFeatures.shape[1])
    return MidTermFeatures.copy()  # , stFeatures,stLabels


    # [MidTermLabels, stLabels] = mtLabelExtraction(gt_labels, Fs, round(mtWin * Fs), round(mtStep * Fs),
    #                                               round(Fs * stWin), round(Fs * stStep))
    #
    # return MidTermFeatures.copy(), MidTermLabels  # , stFeatures,stLabels


























def MidTermSplit(datapath,outFolderName):
    print(datapath)
    fileList = sorted(glob.glob(os.path.join(datapath, "*.xlsx")))
    all_feature_vectors = []

    for file in fileList:
        print("file",file)
        df = pd.read_excel(file)
        time_data = df.iloc[:, 0].to_numpy()
        user = file[-7:-5]
        print("user",user)
        label_file = f'.\\label\\PhysioPalpation_Subject{user}.xlsx'
        label_df = read_label(label_file)
        print("label_df",label_df)
        for i in range(1, 10):
            sensor_data = df.iloc[:, i]
            # print("sensor_data",sensor_data)
            filtered_sensor_data = signal.medfilt(sensor_data, kernel_size=11)
            feature_vectors = featureExtraction(filtered_sensor_data, time_data, 60,30,7.5,7.5)
            feature_vectors = feature_vectors.T
            print("feature_vectors", feature_vectors.shape)
            res_df = pd.DataFrame()
            for idx, feature_vector in enumerate(feature_vectors):
                labels = label_df.iloc[i - 1, :8].to_numpy()
                labeled_feature_vector = np.hstack((feature_vector, labels[idx]))  # Append label for this feature vector
                # Create a temporary DataFrame for this feature vector and append to res_df
                temp_df = pd.DataFrame([labeled_feature_vector])
                res_df = pd.concat([res_df, temp_df], ignore_index=True)

            output_filename = f'{user}_sensor{i}_labelled.csv'
            output_path = os.path.join(outFolderName, output_filename)
            # Ensure the directory exists
            os.makedirs(os.path.join(outFolderName), exist_ok=True)
            # Save the DataFrame to CSV
            res_df.to_csv(output_path, index=False,header=False)
            print(f'Saved data for sensor {i} to {output_path}')






if __name__ == '__main__':
    MidTermSplit('.\\Subject', '.\\preprocessed')