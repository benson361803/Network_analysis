import numpy as np
import os
from tensorflow.keras.utils import to_categorical
import _pickle as cPickle
import matplotlib.pyplot as plt
import itertools
import tensorflow.keras.backend as tfback
import tensorflow as tf
import pywt
from util import util
from sklearn.metrics import roc_curve, auc
class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs=None):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs=None):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


def normalize(eegRaw, normal_all_map=0):
    if len(eegRaw.shape) == 3:
        if normal_all_map == 1:
            print(type(eegRaw))
            normalize_eeg_data = np.zeros((eegRaw.shape), float)
            for u in range(eegRaw.shape[0]):
                minn = np.min(eegRaw[u])
                maxx = np.max(eegRaw[u])
                normalize_eeg_data[u] = (eegRaw[u] - minn) / (maxx - minn)
            del minn
            del maxx
        else:
            print(type(eegRaw))
            normalize_eeg_data = np.zeros((eegRaw.shape), float)
            for u in range(eegRaw.shape[0]):
                for y in range(eegRaw.shape[1]):
                    minn = np.min(eegRaw[u, y, :])
                    maxx = np.max(eegRaw[u, y, :])
                    normalize_eeg_data[u, y, :] = (eegRaw[u, y, :] - minn) / (maxx - minn)
            del minn
            del maxx
    else:
        normalize_eeg_data = np.zeros((eegRaw.shape), float)
        for y in range(eegRaw.shape[1]):
            minn = np.min(eegRaw[y, :])
            maxx = np.max(eegRaw[y, :])
            normalize_eeg_data[y, :] = (eegRaw[y, :] - minn) / (maxx - minn)
        del minn
        del maxx

    return normalize_eeg_data


def normalize_channel_first(left_eegdata, right_eegdata):
    left_normalize = normalize(left_eegdata)
    right_normalize = normalize(right_eegdata)
    trainData = np.zeros(
        (left_eegdata.shape[0] + right_eegdata.shape[0], 1, left_eegdata.shape[1], left_eegdata.shape[2]), float)
    trainData[0:left_eegdata.shape[0], 0] = left_normalize
    trainData[left_eegdata.shape[0]:, 0] = right_normalize
    zero = np.zeros((left_eegdata.shape[0], 1))
    one = zero + 1
    trainLabels = np.vstack((zero, one)).reshape((left_eegdata.shape[0] + right_eegdata.shape[0],))
    Y = to_categorical(trainLabels)

    return trainData, Y


def Trial_Cannel_data(left_eegdata, right_eegdata):
    left_normalize = normalize(left_eegdata)
    right_normalize = normalize(right_eegdata)
    trainData = np.zeros(
        (left_eegdata.shape[0] + right_eegdata.shape[0], left_eegdata.shape[1], left_eegdata.shape[2]), float)
    trainData[0:left_eegdata.shape[0]] = left_normalize[:]
    trainData[left_eegdata.shape[0]:] = right_normalize[:]
    zero = np.zeros((left_eegdata.shape[0], 1))
    zero2 = np.zeros((right_eegdata.shape[0], 1))
    one = zero2 + 1
    Y = np.vstack((zero, one)).reshape((left_eegdata.shape[0] + right_eegdata.shape[0],))

    return trainData, Y


def eegnet_3class_normalize_channel_first(left, right, Rest_left, Rest_right):
    Rest_eegdata_all = np.zeros((Rest_right.shape[0] * 2, Rest_right.shape[1], Rest_right.shape[2]), float)
    Rest_eegdata_all[0:Rest_right.shape[0]] = Rest_left
    Rest_eegdata_all[Rest_right.shape[0]:] = Rest_right
    Rest_eegdata_normalize = normalize(Rest_eegdata_all)
    left_normalize = normalize(left)
    right_normalize = normalize(right)
    trainData = np.zeros((left_normalize.shape[0] + right_normalize.shape[0] + Rest_eegdata_all.shape[0], 1, 16,
                          Rest_eegdata_all.shape[2]), float)
    trainData[0:left_normalize.shape[0], 0] = left_normalize
    trainData[right_normalize.shape[0]:right_normalize.shape[0] * 2, 0] = right_normalize
    trainData[right_normalize.shape[0] * 2:, 0] = Rest_eegdata_normalize

    zero = np.zeros((left_normalize.shape[0], 1))
    zero2 = np.zeros((Rest_eegdata_normalize.shape[0], 1))
    one = zero + 1
    two = zero2 + 2
    trainLabels = np.vstack((zero, one, two)).reshape(
        (left_normalize.shape[0] + right_normalize.shape[0] + Rest_eegdata_normalize.shape[0],))
    Y = to_categorical(trainLabels)

    return trainData, Y


def eegnet_2class_normalize_channel_first(left_eegdata, right_eegdata, NOnormalize_flag=0):
    # print(type(left_eegdata))
    if NOnormalize_flag == 0:
        left_eegdata = normalize(left_eegdata)
        right_eegdata = normalize(right_eegdata)
    elif NOnormalize_flag == 1:
        left_eegdata = normalize(left_eegdata, normal_all_map=1)
        right_eegdata = normalize(right_eegdata, normal_all_map=1)
    else:
        pass

    trainData = np.zeros(
        (left_eegdata.shape[0] + right_eegdata.shape[0], 1, left_eegdata.shape[1], left_eegdata.shape[2]), float)
    trainData[0:left_eegdata.shape[0], 0] = left_eegdata
    trainData[left_eegdata.shape[0]:, 0] = right_eegdata
    zero = np.zeros((left_eegdata.shape[0], 1))
    zero2 = np.zeros((right_eegdata.shape[0], 1))
    one = zero2 + 1
    trainLabels = np.vstack((zero, one)).reshape((left_eegdata.shape[0] + right_eegdata.shape[0],))
    Y = to_categorical(trainLabels)

    return trainData, Y






def eegnet_2class_normalize_channel_last(left_eegdata, right_eegdata, NOnormalize_flag=0):
    if NOnormalize_flag == 0:
        left_eegdata = normalize(left_eegdata)
        right_eegdata = normalize(right_eegdata)
    elif NOnormalize_flag == 1:
        left_eegdata = normalize(left_eegdata, normal_all_map=1)
        right_eegdata = normalize(right_eegdata, normal_all_map=1)
    else:
        pass

    trainData = np.zeros(
        (left_eegdata.shape[0] + right_eegdata.shape[0], left_eegdata.shape[1], left_eegdata.shape[2], 1), float)
    trainData[0:left_eegdata.shape[0], :, :, 0] = left_eegdata
    trainData[left_eegdata.shape[0]:, :, :, 0] = right_eegdata
    zero = np.zeros((left_eegdata.shape[0], 1))
    zero2 = np.zeros((right_eegdata.shape[0], 1))
    one = zero2 + 1
    trainLabels = np.vstack((zero, one)).reshape((left_eegdata.shape[0] + right_eegdata.shape[0],))
    Y = to_categorical(trainLabels)

    return trainData, Y


def eegnet_1class_normalize_channel_first(eegdata, index):
    normalize_data = normalize(eegdata)
    trainData = np.zeros(
        (eegdata.shape[0], 1, eegdata.shape[1], eegdata.shape[2]), float)
    trainData[:, 0] = normalize_data
    if index == 0:
        zero = np.zeros((eegdata.shape[0], 1))
        trainLabels = zero.reshape((eegdata.shape[0],))
    else:
        zero = np.zeros((eegdata.shape[0], 1))
        one = zero + 1
        trainLabels = one.reshape((eegdata.shape[0],))

    Y = to_categorical(trainLabels)
    return trainData, Y


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    # global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


def cutting_eeg(path, start_point, end_point, channel_num, lenth):
    global eegdata
    w = 0
    for root, dirs, files in os.walk(path):
        eegdata = np.zeros((len(files) * 5, channel_num, lenth), float)
        for file in files:
            x_train1 = cPickle.load(open(path + str(file), "rb"))
            print(file)
            ww = w * 5
            print(x_train1.shape)
            for q in range(5):
                qq = q * 5000
                eegdata[q + ww, :, :] = x_train1[qq + start_point:qq + end_point, :channel_num].T
            w = w + 1
    return eegdata


def Load_eegpkl_add_rest(path, start_point, end_point, lenth):
    w = 0
    u = 0
    for root, dirs, files in os.walk(path):
        x_train2 = cPickle.load(open(path + str(files[0]), "rb"))
        print(x_train2.shape)
        eegdata = np.zeros((len(files) * 5, x_train2.shape[1], lenth), float)
        Rest_eegdata = np.zeros((len(files) * 4, x_train2.shape[1], lenth), float)
        for file in files:
            x_train1 = cPickle.load(open(path + str(file), "rb"))
            print(file)
            uu = u * 4
            ww = w * 5
            for o in range(4):
                oo = (o + 1) * 5000
                Rest_eegdata[o + uu, :, :] = x_train1[oo:oo + 1000, :].T
            for q in range(5):
                qq = q * 5000
                eegdata[q + ww, :, :] = x_train1[qq + start_point:qq + end_point, :].T
            w = w + 1
            u = u + 1
    return eegdata, Rest_eegdata


def Load(file_path: list, sub_dic: dict, start: int, end: int, step: float, ft_flag=0):
    if ft_flag == 0:
        left_eegdata = Load_eegpkl_sliding_window2(file_path[0], sub_dic, start, end, step)
        right_eegdata = Load_eegpkl_sliding_window2(file_path[1], sub_dic, start, end, step)
        left_eegdata_test = Load_eegpkl_sliding_window2(file_path[2], sub_dic, start, end, step)
        right_eegdata_test = Load_eegpkl_sliding_window2(file_path[3], sub_dic, start, end, step)
    else:
        sub = str(file_path[0]).split('/', 999)[-2]
        util.checkdir_exist(os.path.join(os.getcwd(),'temporary_file', 'FT_map_inputdata_file'))
        ft_map_file = os.path.join(os.getcwd(),'temporary_file', 'FT_map_inputdata_file', sub)

        util.checkdir_exist(ft_map_file)
        try:
            left_eegdata = np.load(os.path.join(ft_map_file, 'left_eegdata.npy'))
            right_eegdata = np.load(os.path.join(ft_map_file,  'right_eegdata.npy'))
            left_eegdata_test = np.load(os.path.join(ft_map_file, 'left_eegdata_test.npy'))
            right_eegdata_test = np.load(os.path.join(ft_map_file, 'right_eegdata_test.npy'))
        except:
            left_eegdata = normalize_time_frequency(file_path[0], sub_dic, start, end, step)
            right_eegdata = normalize_time_frequency(file_path[1], sub_dic, start, end, step)
            left_eegdata_test = normalize_time_frequency(file_path[2], sub_dic, start, end, step)
            right_eegdata_test = normalize_time_frequency(file_path[3], sub_dic, start, end, step)
            np.save(os.path.join(ft_map_file, 'left_eegdata.npy'), left_eegdata)
            np.save(os.path.join(ft_map_file, 'right_eegdata.npy'), right_eegdata)
            np.save(os.path.join(ft_map_file, 'left_eegdata_test.npy'), left_eegdata_test)
            np.save(os.path.join(ft_map_file, 'right_eegdata_test.npy'), right_eegdata_test)
            print('Processed data has been save')

    return left_eegdata, right_eegdata, left_eegdata_test, right_eegdata_test


def _return_arg(subnum: str):
    pass


def Load_eegpkl_sliding_window2(path: str, sub_dic: dict, *args):
    """
    arg0:star(second)
    arg1:end(second)
    arg2:step(second)ex:0.02,0.05
   arg3:smapling_rate(Hz)

    if want sliding window end-start must bigger than 2 second
    if not star-end ==2 it will nosliding

    """
    start_second = args[0]
    # print(start_second, '666666666666666666')
    end_second = args[1]
    for root, dirs, files in os.walk(path):
        x_train2 = cPickle.load(open(os.path.join(path, str(files[0])), "rb"))

        counter = 0

        for file in files:
            x_train1 = cPickle.load(open(os.path.join(path, str(file)), "rb"))
            sampling_rate = sub_dic[str(file).split('_', 999)[0]]
            start_point = start_second * sampling_rate
            lenth = (end_second - start_second) * sampling_rate
            step_size = int(args[2] * sampling_rate)
            step_num_pertrial = int((lenth - 2 * sampling_rate) / step_size) + 1
            allwindownunm = step_num_pertrial * 5

            if counter == 0:
                one_file_buffer = np.zeros((int(allwindownunm), x_train2.shape[1], 1000), float)
                for d in range(5):
                    ori = d * 10 * sampling_rate
                    ww = d * step_num_pertrial
                    for q in range(int(step_num_pertrial)):
                        star = start_point + q * step_size + ori
                        one_file_buffer[q + int(ww), :, :] = x_train1[int(star):int(star) + 1000, :].T
                all_files_buffer = np.zeros((int(allwindownunm) * len(files), x_train2.shape[1], 1000), float)
                all_files_buffer[:allwindownunm] = one_file_buffer
                counter += 1
            else:
                next_point = counter * allwindownunm
                one_file_buffer = np.zeros((int(allwindownunm), x_train2.shape[1], 1000), float)

                for d in range(5):
                    ori = d * 10 * sampling_rate
                    ww = d * step_num_pertrial
                    for q in range(int(step_num_pertrial)):
                        star = start_point + q * step_size + ori
                        one_file_buffer[q + int(ww), :, :] = x_train1[int(star):int(star) + 1000, :].T
                all_files_buffer[next_point:next_point + allwindownunm] = one_file_buffer
                counter += 1

    return all_files_buffer


def Load_eegpkl_sliding_window(path: str, *args: int):
    """
    arg0:star(second)
    arg1:end(second)
    arg2:step(second)ex:0.02,0.05
    arg3:smapling_rate(Hz)

    if want sliding window star-end must bigger than 2 second
    if not star-end ==2 it will nosliding

    """
    start_second = args[0]
    # print(start_second, '666666666666666666')
    end_second = args[1]

    start_point = start_second * args[3]
    lenth = (end_second - start_second) * args[3]
    step_size = args[2] * args[3]
    step_num_pertrial = int((lenth - 2 * args[3]) / step_size) + 1
    allwindownunm = step_num_pertrial * 5
    fffff = 0
    for root, dirs, files in os.walk(path):
        x_train2 = cPickle.load(open(os.path.join(path, str(files[0])), "rb"))
        print(x_train2.shape)
        pp = np.zeros((len(files) * int(allwindownunm), x_train2.shape[1], 1000), float)
        for file in files:
            # subnum=file.split('_',999)[0]
            ff = fffff * int(allwindownunm)
            x_train1 = cPickle.load(open(os.path.join(path, str(file)), "rb"))
            for d in range(5):
                ww = d * step_num_pertrial
                ori = d * 10 * args[3]
                for q in range(int(step_num_pertrial)):
                    star = start_point + q * step_size + ori
                    pp[q + int(ww) + ff, :, :] = x_train1[int(star):int(star) + 1000, :].T
            fffff += 1

    return pp


'''####&*GUIL'''


def normalize_time_frequency(path: str, sub_dic: dict, *args):
    """
    arg0:star(second)
    arg1:end(second)
    arg2:step(second)ex:0.02,0.05
   arg3:smapling_rate(Hz)

    if want sliding window end-start must bigger than 2 second
    if not star-end ==2 it will nosliding

    """
    wavelet = pywt.wavelist(kind='continuous')[7]
    start_second = args[0]
    # print(start_second, '666666666666666666')
    end_second = args[1]
    for root, dirs, files in os.walk(path):
        x_train2 = cPickle.load(open(os.path.join(path, str(files[0])), "rb"))

        counter = 0

        for file in files:
            x_train1 = cPickle.load(open(os.path.join(path, str(file)), "rb"))
            sampling_rate = sub_dic[str(file).split('_', 999)[0]]
            start_point = start_second * sampling_rate
            lenth = (end_second - start_second) * sampling_rate
            step_size = int(args[2] * sampling_rate)
            step_num_pertrial = int((lenth - 2 * sampling_rate) / step_size) + 1
            allwindownunm = step_num_pertrial * 5

            if counter == 0:
                one_file_buffer = np.zeros((int(allwindownunm), x_train2.shape[1], 1000), float)
                for d in range(5):
                    ori = d * 10 * sampling_rate
                    ww = d * step_num_pertrial
                    for q in range(int(step_num_pertrial)):
                        star = start_point + q * step_size + ori
                        one_file_buffer[q + int(ww), :, :] = x_train1[int(star):int(star) + 1000, :].T
                one_file_buffer = normalize(one_file_buffer)
                fivetrial_F_T = EEG_wavelet_transform(one_file_buffer, wavelet, sampling_rate)
                all_files_buffer = fivetrial_F_T
                counter += 1
            else:
                one_file_buffer = np.zeros((int(allwindownunm), x_train2.shape[1], 1000), float)

                for d in range(5):
                    ori = d * 10 * sampling_rate
                    ww = d * step_num_pertrial
                    for q in range(int(step_num_pertrial)):
                        star = start_point + q * step_size + ori
                        one_file_buffer[q + int(ww), :, :] = x_train1[int(star):int(star) + 1000, :].T
                one_file_buffer = normalize(one_file_buffer)
                fivetrial_F_T = EEG_wavelet_transform(one_file_buffer, wavelet, sampling_rate)
                all_files_buffer = np.concatenate([all_files_buffer, fivetrial_F_T], axis=0)

                counter += 1
    print('Done!!!')

    return all_files_buffer


def Load_eegpkl(path, start_point, end_point, lenth):
    pp = 0
    w = 0
    for root, dirs, files in os.walk(path):
        x_train2 = cPickle.load(open(path + str(files[0]), "rb"))
        print(x_train2.shape)
        pp = np.zeros((len(files) * 5, x_train2.shape[1], lenth), float)
        for file in files:
            x_train1 = cPickle.load(open(path + str(file), "rb"))
            ww = w * 5
            # print(x_train1.shape)
            for q in range(5):
                qq = q * 5000
                pp[q + ww, :, :] = x_train1[qq + start_point:qq + end_point, :].T
            w = w + 1
    print(type(pp))
    return pp


def Docker_Load_eegpkl(path, start_point, end_point, lenth):
    pp = 0
    w = 0
    for root, dirs, files in os.walk(path):
        x_train2 = cPickle.load(open(path + str(files[0]), "rb"))
        print(x_train2.shape)
        pp = np.zeros((len(files) * 5, x_train2.shape[1], lenth), float)
        for file in files:
            x_train1 = cPickle.load(open(path + str(file), "rb"))
            # print(file)
            ww = w * 5
            # print(x_train1.shape)
            for q in range(5):
                qq = q * 5000
                pp[q + ww, :, :] = x_train1[qq + start_point:qq + end_point, :].T
            w = w + 1
    print(type(pp))
    return pp





def plot_roc_curve(true_label, pred_label):
    fpr, tpr, threshold = roc_curve(true_label, pred_label)  ###計算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###計算auc的值
    lw = 2
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率為橫座標，真正率為縱座標做曲線
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    target_names = ['left_hand', 'right_hand']
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)

    # ax = plt.gca()
    # ax.set_xticklabels((ax.get_xticks() + 1).astype(str))
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()


def plot_history_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_history_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def EEG_wavelet_transform(trainData, wavelet, sampling_rate):
    allwindownunm, channel_num = int(trainData.shape[0]), int(trainData.shape[1])

    totalscal = 200
    # fig = plt.figure(1)

    for x in range(allwindownunm):
        for cc in range(channel_num):
            data = trainData[x, cc, :]
            cwtmatr, frequencies = _wavelet_transform(data, wavelet, totalscal, sampling_rate)
            HZ_30 = abs(cwtmatr[:-20:-1])

            if cc == 0:
                allHz_metrix = HZ_30
            else:
                allHz_metrix = np.vstack((allHz_metrix, HZ_30))
        all_metrix = np.expand_dims(allHz_metrix, axis=0)
        if x == 0:
            fivetrial_F_T = all_metrix
        else:
            fivetrial_F_T = np.concatenate([fivetrial_F_T, all_metrix], axis=0)

        # tick_marks = np.arange(len(target_names))
    # ax = fig.add_subplot(111)
    # plt.imshow(fivetrial_F_T[0])
    #
    # plt.xlabel('Time')
    # plt.ylabel('Frequency(Hz)')
    #
    # # ax.xaxis.set_ticks_position('bottom')
    # ax.set(aspect=1.0 / ax.get_data_ratio() * 1)
    # plt.show()

    return fivetrial_F_T


def _wavelet_transform(rawdata: object, wavelet: str, totalscal: int, sampling_rate: int):
    """
        pywt.wavelist(kind='continuous')
        ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']

    """
    fc = pywt.central_frequency(wavelet)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(rawdata, scales, wavelet, 1.0 / sampling_rate)

    return cwtmatr, frequencies
