import src.constants as cst

HP_DLA = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.01, 'min': 0.0001}, # 0.01 in the paper
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32, 64, 256]},  # [32, 64, 128]
    cst.LearningHyperParameter.RNN_HIDDEN.value: {'values': [100]},  # [32, 40, 48]
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [5]},

}

HP_DLA_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 256,
    cst.LearningHyperParameter.RNN_HIDDEN.value: 100,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 5,
}

HP_DLA_LOBSTER_FIXED = {

}