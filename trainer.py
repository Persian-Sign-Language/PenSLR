import warnings
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataset.dataset import DataLoader
from conf import BLANK, OUTLIER_BT, OUTLIER_LT
import numpy as np
from utils.model import output_ctc, calculate_wer_similarity
from path_managers import PathManager
import os
import pickle
from models.general import RegularizationModule
from torchsummary import summary
warnings.filterwarnings('ignore')


class Trainer:
    def __init__(
            self,
            option,
            device="cpu"):
        self.path_manager : PathManager = option.path_manager
        self.dataset_directory = option.dataset_directory
        self.tensorboard_run_name = self.path_manager.summary_writer_path()
        self.visualization = option.visualization
        if self.visualization == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.tensorboard_run_name[0])
        elif self.visualization != 'matplotlib':
            raise ValueError("Visualization parameter only aaccepts these values: ['tensorboard', 'matplotlib']")
        self.save_epoch = option.save_epoch
        self.lr_scheduler = option.lr_scheduler
        self.device = device
        self.optim_fn = option.optim
        self.dataset_directory = option.dataset_directory
        self.features_list = option.features_list
        self.recorders_train = option.recorders_train
        self.recorders_test = option.recorders_test
        self.k_fold_ratio = option.k_fold_ratio
        self.fold_index = option.fold_index
        self.train_val_ratio = option.train_val_ratio
        self.learning_rate = option.learning_rate
        self.train_batch_size = option.train_batch_size
        self.loss_thresh = option.loss_thresh
        self.epoch = option.epoch
        self.regularization_factor = option.regularization_factor
        self.epoch_verbose = option.epoch_verbose
        self.random_seed = option.random_seed
        self.epoch_eval = option.epoch_eval

        self.set_random_seeds()

        self.model : nn.Module = option.M(
            inp=len(self.features_list), out=BLANK
        )
        self.optimizer = None
        self.stats = {
            'loss' : [],
            'word_acc' : [],
            'word_acc_val' : [],
            'sentence_acc' : [],
            'sentence_acc_val' : [],
            'wer_similarity' : [],
            'wer_similarity_val' : [],
            'wer_weighted_similarity' : [],
            'wer_weighted_similarity_val' : []
        }

        self.train_min_word_length = option.train_min_word_length
        self.train_max_word_length = option.train_max_word_length
        self.test_min_word_length = option.test_min_word_length
        self.test_max_word_length = option.test_max_word_length

        self.load()
        
        self.criterion = nn.CTCLoss(blank=BLANK)
        self.criterion.to(self.device)
        self.logs_opt()


    def set_random_seeds(self):
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def load(self): # use path_manager
        model_path, optim_path, dl_path, metrics_path, self.starting_epoch = self.path_manager.find_last_model()
        # Load the model is the path is provided. Else, just move the model to device.
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        else:
            with open(os.path.join(self.path_manager.models_path, 'logs.log'), 'a') as f:
                f.write('------------------------------------------------------------------\n')
                f.write('Summary: \n')
                f.write(str(summary(self.model, (20, len(self.features_list)), verbose=0)))
        self.model.to(self.device)
        # Load optimizer if optim_path is provided. else create a new optimizer.
        if optim_path:
            self.optimizer = self.optim_fn(self.model.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(torch.load(optim_path))
        else:
            self.starting_epoch = 1
            self.optimizer = self.optim_fn(self.model.parameters(), lr=self.learning_rate)
        # load the data loaders and outlier mins and outlier maxs if dl_path is provided.
        if dl_path:
            with open(dl_path, "rb") as f:
                self.dl_train, self.dl_test, self.mins, self.maxs = pickle.load(f)
        else:
            # Else create data loaders, remove outliers and normalize data.
            self.dl_train = DataLoader(self.dataset_directory, self.recorders_train, features=self.features_list, ratio=self.train_val_ratio, min_word_length=self.train_min_word_length, max_word_length=self.train_max_word_length)
            self.dl_test = DataLoader(self.dataset_directory, self.recorders_test, features=self.features_list, ratio=0, min_word_length=self.train_min_word_length, max_word_length=self.train_max_word_length)
            
            self.dl_train.remove_outliers(OUTLIER_BT, OUTLIER_LT)

            self.dl_train.train_val_split(self.fold_index)
            self.mins, self.maxs = self.dl_train.normalize_data(mode='train')
            self.dl_train.normalize_data(mode='val', min_max=(self.mins, self.maxs))
            self.dl_test.normalize_data(mode='test', min_max=(self.mins, self.maxs))
            self.dl_test.train_val_split(0)
        
        if metrics_path:
            with open(metrics_path, "rb") as f:
                self.stats = pickle.load(f)

    
    def save(self, epoch):
        # Make sure that epoch number has 6 digits
        ep = str(epoch).zfill(6)
        # Setting paths for model, optimizer, dataloader, and metrics files
        model_path = os.path.join(self.path_manager.models_path, f"M_{ep}.pth")
        optim_path = os.path.join(self.path_manager.models_path, f"O_{ep}.pth")
        dl_path = os.path.join(self.path_manager.models_path, f"XY.dl")
        metrics_path = os.path.join(self.path_manager.models_path, f"METRICS_{ep}.met")
        # Saving the model, optimizer, dataloader, and metrics files
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)
        with open(dl_path, "wb") as f:
            pickle.dump((self.dl_train, self.dl_test, self.mins, self.maxs), f)
        with open(metrics_path, "wb") as f:
            pickle.dump(self.stats, f)

    def evaluate(self, dl: DataLoader, is_train=True, return_answers=False):
        word_acc, word_tot, sentence_acc, sentence_tot = 0, 0, 0, 0
        L = len(dl.train) if is_train else len(dl.val)
        sentence_tot = L
        total_wer = 0
        total_weighted_wer_sim = 0
        mistakes = []
        length_errors = []
        answers = []
        answers2 = []
        for l in range(L):
            X, y, _, _ = dl.extract_batch(l, 1, self.device, is_train)
            self.model.eval()
            # answers2.append(s)
            with torch.no_grad():
                yp = self.model(X)
                yp = nn.functional.log_softmax(yp, dim=2)
                for j in range(1):
                    ymax = torch.argmax(yp[:, j, :], dim=1)
                    ymax = list(ymax)
                    s = ""
                    for i in ymax:
                        if i == BLANK:
                            s = s + "_"
                        else:
                            s = s + chr(ord('A') + int(i))

                    ans = [str(int(e)) for e in y[j]]
                    ans = [a for a in ans if a != str(BLANK)]
                    s = output_ctc(s)
                    answers.append(s)
                    s = list(map(lambda x: str(ord(x) - ord('A')), list(s)))
                    sentence_acc += ans == s
                    if ans != s:
                        mistakes.append((l, ans, s))
                        length_errors.append(len(ans) - len(s) if len(ans) - len(s) >= 0 else -1)

                    word_tot += len(ans)
                    for cahr in range(len(ans)):
                        if cahr >= len(s):
                            break
                        word_acc += s[cahr] == ans[cahr]
                    # calculate WER
                    sentence_length = len(ans)
                    ans = ' '.join(ans)
                    s = ' '.join(s)
                    wer_sim = calculate_wer_similarity(s, ans)
                    if ans != s:
                        mistakes[-1] = mistakes[-1] + (wer_sim,)
                    total_weighted_wer_sim += wer_sim * sentence_length
                    total_wer += wer_sim
        result = {"word_acc": word_acc / word_tot , "sentence_acc": sentence_acc / sentence_tot, "wer_similarity": total_wer / sentence_tot, "wer_weighted_similarity": total_weighted_wer_sim / word_tot}
        if return_answers:
            result['answers'] = answers
            result['before_ctc'] = answers2
        return result
    
    def test(self):
        train_stats = self.evaluate(self.dl_train, is_train=True)
        val_stats = self.evaluate(self.dl_train, is_train=False)
        test_stats = self.evaluate(self.dl_test, is_train=True)
        return train_stats, val_stats, test_stats
    
    def train(self):
        # Setting the optimizer
        if self.optimizer is None: self.optimizer = self.optim_fn(self.model.parameters(), lr=self.learning_rate)
        # Train Loop
        pbar = tqdm(range(self.starting_epoch, self.epoch + 1))
        for ep in pbar:
            # Shuffle the training set before starting the epoch
            if ep > 1: self.dl_train.shuffle(is_train=True)
            # Set model mode to train
            self.model.train()
            # Iterating over training data batches (The final incomplete batch will be dropped)
            loss_value = 0
            for b in range(len(self.dl_train.train) // self.train_batch_size):
                # Extract input and labels from the training data loader
                X, y, target_lengths, bs = self.dl_train.extract_batch(b, self.train_batch_size, device=self.device, is_train=True)
                # Calculating loss
                self.optimizer.zero_grad()
                yp = self.model(X)
                yp = nn.functional.log_softmax(yp, dim=2)
                input_lengths = torch.LongTensor([yp.shape[0]] * bs).to(self.device).to(torch.int64)
                # calculate loss (perform regularization if needed)
                # print(yp.dtype, y.dtype, input_lengths.dtype, target_lengths.dtype)
                
                loss = self.criterion(yp, y, input_lengths, target_lengths)
                if isinstance(self.model, RegularizationModule):
                    loss = loss + self.regularization_factor * self.model.regularization_loss()
                # Backpropagate and update the weights
                loss.backward()
                self.optimizer.step()
                # Add batch loss to total training loss
                loss_value += loss.item()
            # Append the training loss to the list and update tensorboard
            self.stats['loss'].append(loss_value / (len(self.dl_train.train) // self.train_batch_size))
            if self.visualization == 'tensorboard':
                self.writer.add_scalar('train/loss', self.stats['loss'][-1], ep)
                self.writer.add_scalar('train/lr', self.learning_rate, ep)
            # Update the learning rate with respect to the defined learning rate scheduler
            self.learning_rate = self.lr_scheduler(self.learning_rate, ep, self.stats['loss'])
            self.optimizer.param_groups[0]["lr"] = self.learning_rate
            # Set model mode to evaluation
            self.model.eval()
            # Iterating over validation data batches (The final incomplete batch will be dropped)
            if ep % self.epoch_eval == 0:
                # Adding the new values of evaluation metrics for training data
                train_logs = self.evaluate(self.dl_train, is_train=True)
                for key in train_logs:
                    self.stats[key].append(train_logs[key])
                # Adding the new values of evaluation metrics for validation data
                val_logs = self.evaluate(self.dl_train ,is_train=False)
                for key in val_logs:
                    self.stats[f'{key}_val'].append(val_logs[key])
                # Updating tensorboard summary writer
                if self.visualization == 'tensorboard':
                    for key in train_logs:
                        self.writer.add_scalar(f'train/{key}', train_logs[key], ep)
                        self.writer.add_scalar(f'val/{key}', val_logs[key], ep)
                        

            if ep % self.epoch_verbose == 0:
                # Update tqdm bar with obtained evaluation values
                pbar.set_description(f'Epoch: {ep}, Train Loss: {self.stats["loss"][-1]:.5f}, Train WER: {self.stats["wer_similarity"][-1]:.5f}, Val WER: {self.stats["wer_similarity_val"][-1]:.5f}, lr: {self.optimizer.param_groups[0]["lr"]}')
            
            # Save model, optimizer and dataloader every {self.save_epoch} epochs (or in final epoch)
            if ep % self.save_epoch == 0 or self.stats['loss'][-1] < self.loss_thresh or ep == self.epoch:
                self.save(ep)

                # Save matplotlib images as needed
                if self.visualization == 'matplotlib':
                    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
                    # plot losses
                    ax[0].plot(self.stats['loss'])
                    ax[0].set_xlabel('Epoch')
                    ax[0].set_ylabel('Loss')
                    # plot other stats
                    keys = list(self.stats.keys())
                    for i in range(1, len(keys), 2):
                        ax[i // 2 + 1].plot(self.stats[keys[i]], label=f'train/{keys[i]}')
                        ax[i // 2 + 1].plot(self.stats[keys[i+1]], label=f'val/{keys[i]}')
                        ax[i // 2 + 1].set_xlabel('Epoch')
                        ax[i // 2 + 1].set_ylabel('Accuracy')
                        ax[i // 2 + 1].legend(loc='upper left')
                    # Save figure
                    fig.savefig(f'{self.path_manager.visualization_path}/Result{self.starting_epoch}-{ep}.png') 
                    plt.close(fig)
            
            if self.stats['loss'][-1] < self.loss_thresh:
                break
        print("##################################################################")
        with open(os.path.join(self.path_manager.models_path, 'logs.log'), 'a') as f:
            f.write("Final Stats:\n")
            for key in self.stats:
                print(f'{key}:', self.stats[key][-1])
                f.write(f'{key}: {self.stats[key][-1]}\n')
            f.write("##################################################################\n\n")
            print("##################################################################")

    def logs_opt(self):
        with open(os.path.join(self.path_manager.models_path, 'logs.log'), 'a') as f:
            print("Starting epoch       :", int(self.starting_epoch))
            print("Dataset Directory    :", self.dataset_directory)
            print("Fold index           :", self.fold_index)
            print("Num Of Training Set  :", len(self.dl_train.train))
            print("Num Of Validation Set:", len(self.dl_train.val))
            print("Num Of Test Set      :", len(self.dl_test.train))
            print("Num Of Parameters    :", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            print("Optimizer Function   :", self.optim_fn.__name__)
            print("Learning Rate        :", self.learning_rate)
            print("Tensorboard Exp Name :", self.tensorboard_run_name[1])
            print("Train Recorders      :", " ".join(self.recorders_train))
            print("Test Recorders       :", " ".join(self.recorders_test))
            print("Batch Size           :", self.train_batch_size)
            print("Latest loss          :", self.stats['loss'][-1] if len(self.stats['loss']) != 0 else None)
            print("Visualization Method :", self.visualization)
            print("##################################################################")

            f.write(f"Starting epoch       : {int(self.starting_epoch)}\n")
            f.write(f"Dataset Directory    :{self.dataset_directory}\n")
            f.write(f"Fold index           :{self.fold_index}\n")
            f.write(f"Num Of Training Set  : {len(self.dl_train.train)}\n")
            f.write(f"Num Of Validation Set: {len(self.dl_train.val)}\n")
            f.write(f"Num Of Test Set      : {len(self.dl_test.train)}\n")
            f.write(f"Num Of Parameters    : {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}\n")
            f.write(f"Optimizer Function   : {self.optim_fn.__name__}\n")
            f.write(f"Learning Rate        : {self.learning_rate}\n")
            f.write(f"Tensorboard Exp Name : {self.tensorboard_run_name[1]}\n")
            f.write(f"Train Recorders      : {' '.join(self.recorders_train)}\n")
            f.write(f"Test Recorders       : {' '.join(self.recorders_test)}\n")
            f.write(f"Batch Size           : {self.train_batch_size}\n")
            f.write(f"Latest loss          : {self.stats['loss'][-1] if len(self.stats['loss']) != 0 else None}\n")
            f.write(f"Visualization Method : {self.visualization}\n")
            f.write("##################################################################\n\n")

    def get_feature_importance(self, repeat = 5):
        _, _, test_stats = self.test()
        stats_difference = {}
        for i in tqdm(range(len(self.features_list))):
            feature = self.features_list[i]
            stats_difference[feature] = {}
            for _ in range(repeat):
                permutated_dl = self.dl_test.shuffle_feature(feature)
                new_stats = self.evaluate(permutated_dl, is_train=True)
                for key in test_stats:
                    stats_difference[feature][key] = 0
                for key in test_stats:
                    increase_percent = (new_stats[key] - test_stats[key]) / test_stats[key] * 100
                    stats_difference[feature][key] += np.round(increase_percent / repeat, 4)
        return stats_difference
            