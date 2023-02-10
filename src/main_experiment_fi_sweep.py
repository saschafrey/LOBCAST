
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *


def experiment_FI(kset, now=None, models_todo=None, servers=None):

    now, server_name, server_id, n_servers = experiment_preamble(now, servers)

    models_todo = models_todo if models_todo is not None else cst.Models
    for mod in list(models_todo)[server_id::n_servers]:
        for k in kset:
            print("Running FI experiment on {}, with K={}".format(mod, k))

            try:
                cf: Configuration = Configuration(now)
                set_seeds(cf)

                cf.CHOSEN_DATASET = cst.DatasetFamily.FI
                cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
                cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI
                cf.CHOSEN_PERIOD = cst.Periods.FI
                cf.CHOSEN_MODEL = mod

                cf.IS_WANDB = 1
                cf.IS_TUNE_H_PARAMS = True

                cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value
                launch_wandb(cf)

            except KeyboardInterrupt:
                print("There was a problem running on", server_name.name, "FI experiment on {}, with K={}".format(mod, k))
                sys.exit()


servers = [cst.Servers.ALIEN1]
models_todo = [cst.Models.MLP]
kset = [cst.FI_Horizons.K2]
now = "BINCTABL, ATNBoF, K1, K2, sweep"

experiment_FI(kset, servers=servers, models_todo=None, now=now)
