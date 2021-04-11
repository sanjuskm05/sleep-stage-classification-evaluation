import glob
import math
import ntpath
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from mne.io import read_raw_edf
from torch.utils.data import TensorDataset

import dhedfreader
# Label values
from MintNet import MintNet

W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30
# data_dir="/Users/pradeep/Desktop/mcs/cs598DLhealthcare/project/sleep-edf-database-expanded-1.0.0/sleep-cassette/"
data_dir = "C:/Users/omesha/Documents/Illinois/sleepdata/Sleep-Stage-Classification-master/Sleep-Stage-Classification-master/physionet-sleep-data"
# output_dir="/Users/pradeep/Desktop/mcs/cs598DLhealthcare/project/EEGFPzCz/"
output_dir = data_dir + "/out"
psg_fnames = glob.glob(os.path.join(data_dir, "*PSG.edf"))
ann_fnames = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))
psg_fnames.sort()
ann_fnames.sort()
psg_fnames = np.asarray(psg_fnames)
ann_fnames = np.asarray(ann_fnames)
##['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']
select_ch = 'EEG Fpz-Cz'


##Below is working code corrected

def getBatch(number_of_subj, output_dir):
    npz_files = sorted(glob.glob(os.path.join(output_dir, "*.npz")))
    X_data = []
    Y_data = []
    for fn in npz_files[:number_of_subj]:
        samples = np.load(fn)
        X_data.extend(samples['x'])
        Y_data.extend(samples['y'])
    return (X_data, Y_data)


print("++++++++++++++")

##['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']
select_ch = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal']
# output_dirnew="/Users/pradeep/Desktop/mcs/cs598DLhealthcare/project/EEGFPzCz/test/"
output_dirnew = data_dir + "/out/test"
#####Multichannel code
for i in range(len(psg_fnames)):
    if not "SC4001E0-PSG.edf" in psg_fnames[i]:
        continue
    # existfileorNot = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
    # if(os.path.exists(os.path.join(output_dirnew, existfileorNot))):
    #     continue

    raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
    sampling_rate = raw.info['sfreq']
    raw_ch_df = raw.to_data_frame(scalings=100)[select_ch]
    # raw_ch_df = raw_ch_df.to_frame()
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))

    # Get raw header
    f = open(psg_fnames[i], 'r', encoding='iso-8859-1')
    reader_raw = dhedfreader.BaseEDFReader(f)
    reader_raw.read_header()
    h_raw = reader_raw.header

    f.close()
    raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

    # Read annotation and its header
    f = open(ann_fnames[i], 'r', encoding='iso-8859-1')
    reader_ann = dhedfreader.BaseEDFReader(f)
    reader_ann.read_header()
    h_ann = reader_ann.header
    _, _, ann = list(zip(*reader_ann.records()))
    f.close()
    ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

    # Assert that raw and annotation files start at the same time
    assert raw_start_dt == ann_start_dt

    # Generate label and remove indices
    remove_idx = []  # indicies of the data that will be removed
    labels = []  # indicies of the data that have labels
    label_idx = []
    for a in ann[0]:
        onset_sec, duration_sec, ann_char = a
        ann_str = "".join(ann_char)
        label = ann2label[ann_str]
        if label != UNKNOWN:
            if duration_sec % EPOCH_SEC_SIZE != 0:
                raise Exception("Something wrong")
            duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)
            idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
            label_idx.append(idx)

            print("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        else:
            idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
            remove_idx.append(idx)

            print("Remove onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
    labels = np.hstack(labels)

    print("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
    if len(remove_idx) > 0:
        remove_idx = np.hstack(remove_idx)
        select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
    else:
        select_idx = np.arange(len(raw_ch_df))
    print("after remove unwanted: {}".format(select_idx.shape))

    # Select only the data with labels
    print("before intersect label: {}".format(select_idx.shape))
    label_idx = np.hstack(label_idx)
    select_idx = np.intersect1d(select_idx, label_idx)
    print("after intersect label: {}".format(select_idx.shape))

    # Remove extra index
    if len(label_idx) > len(select_idx):
        print("before remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
        extra_idx = np.setdiff1d(label_idx, select_idx)
        # Trim the tail
        if np.all(extra_idx > select_idx[-1]):
            n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
            n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
            select_idx = select_idx[:-n_trims]
            labels = labels[:-n_label_trims]
        print("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))

    if (select_idx.shape[0] == 0):
        continue
    # Remove movement and unknown stages if any
    raw_ch = raw_ch_df.values[select_idx]

    # Verify that we can split into 30-s epochs
    if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
        raise Exception("Something wrong")
    n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

    # Get epochs and their corresponding labels
    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)

    assert len(x) == len(y)

    # Select on sleep periods
    w_edge_mins = 30
    nw_idx = np.where(y != stage_dict["W"])[0]
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    if start_idx < 0: start_idx = 0
    if end_idx >= len(y): end_idx = len(y) - 1
    select_idx = np.arange(start_idx, end_idx + 1)
    print(("Data before selection: {}, {}".format(x.shape, y.shape)))
    x = x[select_idx]
    y = y[select_idx]
    print(("Data after selection: {}, {}".format(x.shape, y.shape)))

    # Save
    filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
    save_dict = {
        "x": x,
        "y": y,
        "fs": sampling_rate,
        "ch_label": select_ch,
        "header_raw": h_raw,
        "header_annotation": h_ann,
    }
    np.savez(os.path.join(output_dirnew, filename), **save_dict)

    print("\n=======================================\n")


def getBatch(number_of_subj, output_dirnew):
    npz_files = sorted(glob.glob(os.path.join(output_dirnew, "*.npz")))
    X_data = []
    Y_data = []
    for fn in npz_files[:number_of_subj]:
        samples = np.load(fn)
        X_data.extend(samples['x'])
        Y_data.extend(samples['y'])
    return (X_data, Y_data)


from torch.utils.data import Dataset


class EEGDataset(Dataset):

    def __init__(self, a):
        """
        TODO: init the Dataset instance.
        """
        self.X = a[0]
        self.Y = a[1]

    def __len__(self):
        """
        TODO: Denotes the total number of samples
        """

        return len(self.Y)

    def __getitem__(self, i):
        return (self.X[i],self.Y[i])
        #return (self.X[i][:,0],self.Y[i])


def load_data(dataset, batch_size=32):
    """
    Return a DataLoader instance basing on a Dataset instance, with batch_size specified.
    Note that since the data has already been shuffled, we set shuffle=False
    """

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_model(model, train_loader, n_epoch=5, lr=0.003, device=None):
    import torch.optim as optim
    """
    Comments goes here
    """
    device = device or torch.device('cpu')
    model.train()
    loss_history = []
    lossFunc = nn.CrossEntropyLoss()
    # lossFunc = nn.MSELoss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epoch):
        curr_epoch_loss = []
        accuracy = MeanMeasure()
        for X, Y in train_loader:
            optimizer.zero_grad()
            X = X.to(device)
            Y = Y.to(device)
            Y_hat = model(X, device).float()
            Y = Y.long()
            loss = lossFunc(Y_hat, Y)
            loss.backward()
            optimizer.step()
            curr_epoch_loss.append(loss.cpu().data.numpy())
            batch_acc = calculate_batch_accuracy(Y_hat, Y)
            accuracy.update(batch_acc, Y.size(0))
        print(f"epoch{epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
        loss_history += curr_epoch_loss
        print(datetime.now().strftime("%Y-%m-%d %H:%M"), "batch_acc->", accuracy.average)


        ## Evaluating on every epoch
        #pred, truth = eval_model(model, test_loader, device=device)
        #auroc, f1 = evaluate_predictions(truth, pred)
        #print(f"AUROC={auroc} and F1={f1}")
    return model, loss_history


def eval_model(model, dataloader, device=None):
    """
    Comments goes here
    """
    #device = device or torch.device('cpu')
    model.eval()
    pred_all = []
    Y_test = []
    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)
        Y_hat = model(X, device).float()
        pred_all.append(Y_hat.cpu().detach().numpy())
        Y_test.append(Y.cpu().detach().numpy())
    pred_all = np.concatenate(pred_all, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    return pred_all, Y_test

def evaluate_predictions(truth, pred):
    """
    TODO: Evaluate the performance of the predictoin via AUROC, and F1 score

    each prediction in pred is a vector representing [p_0, p_1].
    When defining the scores we are interesed in detecting class 1 only
    (Hint: use roc_auc_score and f1_score from sklearn.metrics)
    return: auroc, f1
    """
    from sklearn.metrics import roc_auc_score, f1_score

    pred = np.argmax(pred, axis=1)
    auroc = roc_auc_score(truth, pred, multi_class='ovo')
    f1 = f1_score(truth, pred)

    return auroc, f1

def calculate_batch_accuracy(output, target):
    with torch.no_grad():

        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

        return correct * 100.0 / batch_size

class MeanMeasure(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.itemValue = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, itemValue, n=1):
        self.itemValue = itemValue
        self.sum += itemValue * n
        self.count += n
        self.average = self.sum / self.count

##############

# print('__Python VERSION:', sys.version)
# print('__pyTorch VERSION:', torch.__version__)
# print('__CUDA VERSION', )
# from subprocess import call
# # call(["nvcc", "--version"]) does not work
# print('__CUDNN VERSION:', torch.backends.cudnn.version())
# print('__Number CUDA Devices:', torch.cuda.device_count())
# print('__Devices')
# # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# print('Active CUDA Device: GPU', torch.cuda.current_device())
# print ('Available devices ', torch.cuda.device_count())
# print ('Current cuda device ', torch.cuda.current_device())
###############
def main():
    train_loader = load_data(EEGDataset(getBatch(100, output_dirnew)))
    test_loader = load_data(EEGDataset(getBatch(1, output_dirnew)))
    # for a, b in train_loader:
    #     print(a.shape)

    print(torch.cuda.is_available())
    if torch.cuda.is_available():
      dev = "cuda:0"
      torch.cuda.empty_cache()
    else:
      dev = "cpu"
    device = torch.device(dev)
    print(datetime.now().strftime("%Y-%m-%d %H:%M"))
    n_epoch = 50
    lr = .03#0.003

    n_dim=6#number of channels


    model = MintNet(n_dim)
    model = model.to(device)

    model, loss_history = train_model(model, train_loader, n_epoch=n_epoch, lr=lr, device=device)
    pred, truth = eval_model(model, test_loader, device=device)
    #auroc, f1 = evaluate_predictions(truth, pred)
    #print(f"AUROC={auroc} and F1={f1}")

    print(datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == '__main__':
    main()