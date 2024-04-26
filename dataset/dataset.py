from typing import List
import os
import numpy as np
import re
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from conf import *
import copy

class DataLoader:
    def __init__(self, directory: str, recorders, features: list = None, ratio :float = 0.2, min_word_length=1, max_word_length=3) -> None:
        self.classes2index = CLASSES2INDEX
        self.index2classes = INDEX2CLASSES
        self.classes = CLASSES
        self.recorders = recorders
        self.dir = directory
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.data = []
        self.BLANK = len(self.classes)
        self.ratio = ratio
        self.features = np.array([FEATURES2INDEX[f] for f in features])
        self.features_realtive_index = {k:v for v,k in enumerate(features)}
        self.splited = False
        self.stats = {r:{} for r in recorders}
        self.data_loader()

    def line2float(self, line: str):
        """
        extract select features from readed line to numpy.ndarray
        """
        if self.features is None:
            return np.array([float(i) for i in line[:-2].split(" ")])
        return np.array([float(i) for i in line[:-2].split(" ")])[self.features]
    
    def read_file(self, file:str):
        """
        read data from file
        """
        with open(file, "r") as f:
            lines = f.readlines()
        x = [[]]
        for i in range(len(lines)):
            if ";" in lines[i]:
                x.append([])
            else:
                x[-1].append(self.line2float(lines[i]))
        if x[-1] == []:
            return x[:-1]
        return x

    def filename2label(self, file_path:str):
        """
        read label from file name for example AbiAst.txt -> [Abi, Ast] -> [index(Abi), index(Ast)]
        """
        y = file_path.split("/")[-1].split(".")[0] if "/" in file_path else file_path.split(".")[0]
        y = re.findall('[A-Z][^A-Z]*', y)
        label = [self.classes2index[move] for move in y]
        return label

    def file2data(self, file_path: str):
        # it is just for better experience
        X = self.read_file(file=file_path)
        return X
    
    def data_loader(self):
        """
        This function load all possible for selected recorder
        warning: this shuffles data
        """
        for recorder in self.recorders:
            files = os.listdir(os.path.join(self.dir, recorder))
            for file in files:
                if file.endswith("txt"):
                    file_path = os.path.join(self.dir, recorder, file)
                    label = self.filename2label(file_path)
                    if not self.min_word_length <= len(label) <= self.max_word_length:
                        continue
                    count = self.stats[recorder].get(f'{len(label)}', 0)
                    X = self.file2data(file_path)
                    self.stats[recorder][f'{len(label)}'] = count + len(X)
                    for x in X:
                        self.data.append([np.array(x), label])
                    
        np.random.shuffle(self.data)
    
    def train_val_split(self, ith):
        """
        just split data using ratio and then set ith part to val and other part to train
        """
        split_size = int(self.ratio * len(self.data))
        point1 = ith * split_size
        point2 = point1 + split_size
        self.val = self.data[point1:point2]
        self.train = self.data[:point1] + self.data[point2:]
        self.splited = True
    
    def pad_x(self, x, length):
        """
        for batching data we need to pad all data in a batch to same length
        """
        return np.pad(x, ((0, length - x.shape[0]), (0, 0)), mode='constant', constant_values = 0.)

    def pad_y(self, y, length):
        """
        for batching data we need to pad all data in a batch to same length
        """
        out = [i for i in y]
        for i in range(length - len(y)):
            out.append(self.BLANK)
        return out

    def extract_batch(self, index, bs, device="cpu", is_train=True):
        """
        extract batch using index and batch size
        it returns X, y, target_lengths, batch_size
        for better understanding read CTCLoss pytorch
        """
        data = self.train if is_train else self.val
        last_index = min(len(data), (index + 1) * bs)
        batch = data[index * bs: last_index]
        max_x = max([d[0].shape[0] for d in batch])
        max_y = max([len(d[1]) for d in batch])
        X = []
        y = []
        target_lengths = []
        for d in batch:
            X.append(self.pad_x(d[0], max_x))
            y.append(self.pad_y(d[1], max_y))
            target_lengths.append(len(d[1]))
        y = np.array(y)
        X = np.array(X)
        X = torch.from_numpy(X).float().to(device)
        y = torch.from_numpy(y).int().to(device)
        target_lengths = torch.from_numpy(np.array(target_lengths)).int().to(device)
        return X, y, target_lengths, len(batch)

    def normalize_data(self, mode = 'train', min_max=None):
        """
        just a function to normalize data set
        """
        if mode not in ['train', 'test', 'val']:
            raise ValueError("mode parameter must be in list ['train', 'test', 'val']")
        
        if mode != 'test' and not self.splited:
            raise ValueError("You must call train_val_split before normalizing the train dataloader.")
        
        if mode != 'train' and not min_max:
            raise ValueError("You must set min_max in val and test mode.")
        
        mins, maxs = None, None
        data = self.train if mode == 'train' else self.val if mode == 'val' else self.data
        if not min_max:
            # finding maxs and mins
            mins = np.ones((1, len(self.features)), dtype=float) * (10 ** 6)
            maxs = np.ones((1, len(self.features)), dtype=float) * (-10 ** 5)

            for sample_label_pair in data:
                sample, _ = sample_label_pair
                data_mins = np.min(sample, axis = 0, keepdims = True)
                data_maxs = np.max(sample, axis = 0, keepdims = True)
                mins = np.minimum(mins, data_mins)
                maxs = np.maximum(maxs, data_maxs)
        else:
            mins, maxs = min_max

        # updating the data
        for sample_label_pair in data:
            sample, _ = sample_label_pair
            sample_np = (np.array(sample) - mins) / (maxs - mins)
            if min_max:
                sample_np[sample_np > 1] = 1
                sample_np[sample_np < 0] = 0
            sample_label_pair[0] = sample_np

        return mins, maxs
    
    def remove_outliers(self, outlier_bt={}, outlier_lt={}):
        """
        just a function to remove outlire from data
        """
        outlier_ids = []
        for id, sample_label_pair in enumerate(self.data):
            sample, _ = sample_label_pair
            for feature_name in outlier_bt:
                index = self.features_realtive_index.get(feature_name, None)
                if index:
                    for i in range(sample.shape[0]):
                        if sample[i, index] > outlier_bt[feature_name]:
                            outlier_ids.append(id)
                            break
            
        for id, sample_label_pair in enumerate(self.data):
            sample, _ = sample_label_pair
            for feature_name in outlier_lt:
                index = self.features_realtive_index.get(feature_name, None)
                if index:
                    for i in range(sample.shape[0]):
                        if sample[i, index] < outlier_lt[feature_name]:
                            outlier_ids.append(id)
                            break
      
        self._delete_items_from_list(self.data, outlier_ids)

    def _delete_items_from_list(self, the_list: list, indexes: List[int]):
        for index in sorted(indexes, reverse=True):
                del the_list[index]
    
    def shuffle(self, is_train=True):
        if is_train:
            np.random.shuffle(self.train)
        else:
            np.random.shuffle(self.val)

    def plot_distributions(self, figsize=(10, 5)):
        indexes = np.arange(len(CLASSES))
        width = 0.8
        if self.splited:
            train_samples_num =  [0 for _ in range(len(CLASSES))]
            for sample_label_pair in self.train:
                for cls in sample_label_pair[1]:
                    train_samples_num[cls] += 1
            val_samples_num =  [0 for _ in range(len(CLASSES))]
            for sample_label_pair in self.val:
                for cls in sample_label_pair[1]:
                    val_samples_num[cls] += 1
            # plotting
            _, ax = plt.subplots(2, 1, figsize=figsize)
            ax[0].bar(indexes, train_samples_num, width=width, color='red')
            ax[0].set_title('Training Set')
            ax[0].set_xticks(indexes, CLASSES)
            for i, num in enumerate(train_samples_num):
                ax[0].text(i, num + 0.01, str(num))
            ax[1].bar(indexes, val_samples_num, width=width, color='blue')
            ax[1].set_title('Validation Set')
            ax[1].set_xticks(indexes, CLASSES)
            for i, num in enumerate(val_samples_num):
                ax[1].text(i, num + 0.01, str(num))
        else:
            plt.figure(figsize=figsize)
            samples_num =  [0 for _ in range(len(CLASSES))]
            for sample_label_pair in self.data:
                for cls in sample_label_pair[1]:
                    samples_num[cls] += 1
            # plotting
            plt.bar(indexes, samples_num, width=width, color='red')
            for i, num in enumerate(samples_num):
                plt.text(i, num + 0.01, str(num))
            plt.xticks(indexes, CLASSES)
        plt.show() 
    
    def _pad_or_trim(self, data, length):
        # perform padding if the requested length is bigger than the length of data
        if length > len(data):
            offset = (length - len(data)) // 2
            output = np.copy(data)
            output = np.pad(output, offset)
            if (length - len(data)) % 2 == 1:
                output = np.append(output, 0)
            return output
        # trim the data if the requested length is smaller than the length of data
        offset = (len(data) - length) // 2
        if (len(data) - length) % 2 == 0:
            return data[offset: len(data) - offset]
        return data[offset + 1: len(data) - offset]

    def shuffle_feature(self, f_name):
        f_index = self.features_realtive_index[f_name]
        new_data_loader = copy.deepcopy(self)
        values = [d[0][:,f_index] for d in new_data_loader.data]
        np.random.shuffle(values)
        for i in range(len(new_data_loader.data)):
            prev_data_len = new_data_loader.data[i][0].shape[0]
            new_data_loader.data[i][0][:, f_index] = self._pad_or_trim(values[i], prev_data_len)
        return new_data_loader
