

from torch.utils import data
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
import src.constants as cst
import collections

from src.data_preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder
from src.config import Configuration


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(
            self,
            config: Configuration,
            dataset_type,
            stocks,
            start_end_trading_day,
            stockName2mu=dict(),
            stockName2sigma=dict(),
            num_classes=3,
            num_snapshots=100,
            one_hot_encoding=False
    ):
        self.config = config
        self.ys_occurrences = None
        self.dataset_type = dataset_type
        self.stocks = stocks
        self.start_end_trading_day = start_end_trading_day
        self.num_snapshots = num_snapshots
        self.num_classes = num_classes

        self.stockName2mu, self.stockName2sigma = stockName2mu, stockName2sigma
        self.sample_size = self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS] # 100

        stockName2databuilder = dict()

        # Choose the stock names to open to build the specific dataset.
        # No need to open all for test set, because mu/sig are pre-computed when prev opened train and dev
        stocksToOpen = None
        if dataset_type == cst.DatasetType.TRAIN:
            # we open also the TEST stock(s) to determine mu and sigma for normalization, needed for all
            stocksToOpen = list(set(config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value + config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value))  # = [LYFT, NVDA]
        elif dataset_type == cst.DatasetType.VALIDATION:
            stocksToOpen = config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value  # = [LYFT]
        elif dataset_type == cst.DatasetType.TEST:
            stocksToOpen = config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value   # = [NVDA]

        for stock in stocksToOpen:
            path = cst.DATASET_LOBSTER + f'_data_dwn_48_332__{stock}_{config.CHOSEN_PERIOD.value["train"][0]}_{config.CHOSEN_PERIOD.value["test"][1]}_10'

            normalization_mean = stockName2mu[stock] if stock in stockName2mu else None
            normalization_std = stockName2sigma[stock] if stock in stockName2sigma else None

            print(dataset_type, '\t', stocks, '\t', stock, '\t', start_end_trading_day, '\t', normalization_mean, '\t', normalization_std, '\t', path)

            databuilder = LOBSTERDataBuilder(
                stock,
                path,
                config=config,
                n_lob_levels=cst.N_LOB_LEVELS,
                dataset_type=dataset_type,
                start_end_trading_day=start_end_trading_day,
                crop_trading_day_by=60*30,
                window_size_forward=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
                window_size_backward=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
                num_snapshots=num_snapshots,
                label_dynamic_scaler=config.HYPER_PARAMETERS[cst.LearningHyperParameter.LABELING_SIGMA_SCALER],
                is_data_preload=config.IS_DATA_PRELOAD
            )

            self.stockName2mu[stock], self.stockName2sigma[stock] = databuilder.normalization_means, databuilder.normalization_stds
            stockName2databuilder[stock] = databuilder  # STOCK: databuilder

        print('stockName2mu:', self.stockName2mu)
        print('stockName2sigma:', self.stockName2sigma)

        Xs, Ys, Ss, ignore_indices_len = list(), list(), list(), [0]
        for stock in self.stocks:
            print("Handling", stock, "for dataset", dataset_type)
            databuilder = stockName2databuilder[stock]

            data_x, data_y = databuilder.get_X_nx40(), databuilder.get_Y_n()
            Xs.append(data_x)
            Ys.append(data_y)
            Ss.extend([stock]*len(data_y))
            ignore_indices_len.append(len(data_y))

        # removes the indices that are the first 100
        ignore_indices = []
        ind_sf = 0
        for iv in range(len(ignore_indices_len)-1):
            p = ind_sf + ignore_indices_len[iv]
            ignore_indices += list(range(p, p+self.sample_size))  # [range(0, 100), range(1223, 1323), ]
            ind_sf = p

        self.x = torch.from_numpy(np.concatenate(Xs, axis=0)).type(torch.FloatTensor)
        self.y = np.concatenate(Ys, axis=0).astype(int)
        self.stock_sym_name = Ss

        self.indexes_chosen = self.__under_sampling(self.y, ignore_indices)
        self.x_shape = (self.sample_size, self.x.shape[1])

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.indexes_chosen)

    def __getitem__(self, index):
        """ Generates samples of data. """

        id_sample = self.indexes_chosen[index]
        x, y, s = self.x[id_sample-self.sample_size:id_sample, :], self.y[id_sample], self.stock_sym_name[id_sample]
        return x, y, s

    def __under_sampling(self, y, ignore_indices):
        """ Discard instances of the majority class. """
        print("Doing under-sampling...")

        y_without_snap = [y[i] for i in range(len(y)) if i not in ignore_indices]  # removes the indices of the first sample for each stock

        occurrences = self.__compute_occurrences(y_without_snap)
        i_min_occ = min(occurrences, key=occurrences.get)  # index of the class with the least instances
        n_min_occ = occurrences[i_min_occ]                 # number of occurrences of the minority class

        occs = np.array([occurrences[k] for k in sorted(occurrences)])
        self.loss_weights = torch.Tensor(occs / np.sum(occs))

        indexes_ignore = set(ignore_indices)
        indexes_chosen = []
        for i in [cst.Predictions.UPWARD.value, cst.Predictions.STATIONARY.value, cst.Predictions.DOWNWARD.value]:
            indexes = np.where(y == i)[0]
            indexes = np.array(list(set(indexes) - indexes_ignore))  # the indices of the first sample for each stock

            assert len(indexes) >= self.config.INSTANCES_LOWER_BOUND, "The instance is not well formed, there are less than {} instances for the class {} ({}).".format(self.config.INSTANCES_LOWER_BOUND, i, len(indexes))
            indexes_chosen += list(self.config.RANDOM_GEN_DATASET.choice(indexes, n_min_occ, replace=False))

        indexes_chosen = np.sort(indexes_chosen)
        return indexes_chosen

    def __compute_occurrences(self, y):
        occurrences = collections.Counter(y)
        return occurrences
