from enum import Enum
import numpy as np
import torch
from datetime import datetime

np.set_printoptions(suppress=True)


class TuningVars(Enum):
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "lr"
    EPOCHS = "epochs"
    IS_SHUFFLE = "is_shuffle"

    MLP_HIDDEN = "hidden_mlp"
    LSTM_HIDDEN = "lstm_hidden"
    LSTM_N_HIDDEN = "lstm_n_hidden"

    BACKWARD_WINDOW = "window_size_backward"
    FORWARD_WINDOW = "window_size_forward"
    LABELING_THRESHOLD = "labeling_threshold"
    LABELING_SIGMA_SCALER = "labeling_sigma_scaler"

    FI_HORIZON = 'fi_horizon_k'

class Metrics(Enum):
    LOSS = 'loss'
    F1 = 'f1'
    PRECISION = 'precision'
    RECALL = 'recall'
    ACCURACY = 'accuracy'


class ModelSteps(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class NormalizationType(Enum):
    Z_SCORE = 0
    DYNAMIC = 1
    NONE = 2


class WinSize(Enum):
    SEC10 = 10
    SEC20 = 20
    SEC30 = 30
    SEC50 = 50
    SEC100 = 100

    MIN01 = 60
    MIN05 = 60 * 5
    MIN10 = 60 * 10
    MIN20 = 60 * 20


class Predictions(Enum):
    UPWARD = 2
    DOWNWARD = 0
    STATIONARY = 1


# to use in the future
class Models(Enum):
    DEEPLOB = "DeepLob"
    MLP = "MLP"
    CNN = "CNN"
    LSTM = "LSTM"
    MLP_FI = "MLP_FI"


class DatasetFamily(Enum):
    FI = "Fi"
    LOBSTER = "Lobster"

class Horizons(Enum):
    K1 = 1
    K2 = 2
    K3 = 3
    K5 = 5
    K10 = 10

class Granularity(Enum):
    """ The possible Granularity to build the OHLC old_data from lob """
    Sec1 = "1S"
    Sec5 = "5S"
    Sec15 = "15S"
    Sec30 = "30S"
    Min1 = "1Min"
    Min5 = "5Min"
    Min15 = "15Min"
    Min30 = "30Min"
    Hour1 = "1H"
    Hour2 = "2H"
    Hour6 = "6H"
    Hour12 = "12H"
    Day1 = "1D"
    Day2 = "2D"
    Day5 = "7D"
    Month1 = "30D"


class OrderEvent(Enum):
    """ The possible kind of orders in the lob """
    SUBMISSION = 1
    CANCELLATION = 2
    DELETION = 3
    EXECUTION = 4
    HIDDEN_EXECUTION = 5
    CROSS_TRADE = 6
    TRADING_HALT = 7
    OTHER = 8


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"


CLASS_NAMES = ["DOWN", "STATIONARY", "UP"]

OHLC_DATA = "old_data/ohlc_data/"
DEVICE = 1 if torch.cuda.is_available() else 0
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_GAN = "data/GAN_models/"


EPOCHS = 150

SEED = 0
RANDOM_GEN_DATASET = np.random.RandomState(SEED)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATE_EVERY = 1
MLP_HIDDEN = 128

LSTM_HIDDEN = 32
LSTM_N_HIDDEN = 1

N_LOB_LEVELS = 10
LABELING_SIGMA_SCALER = .3  # dynamic threshold
BACKWARD_WINDOW = WinSize.SEC10.value
FORWARD_WINDOW  = WinSize.SEC10.value
SAVED_MODEL_DIR = "data/saved_models"

HORIZON = 5

TRAIN_SPLIT_VAL = .7

DATA_SOURCE = "data/"
DATASET_LOBSTER = "AVXL_2021-08-01_2021-08-31_10"
DATASET_FI = "FI-2010/BenchmarkDatasets"
DATA_PICKLES = "data/pickles/"

DATASET_FAMILY = DatasetFamily.LOBSTER

PROJECT_NAME = "lob-adversarial-attacks-22"
IS_WANDB = True

IS_DATA_PRELOAD = False

CHOSEN_MODEL = Models.MLP
IS_SHUFFLE_INPUT = True
INSTANCES_LOWERBOUND = 1000
