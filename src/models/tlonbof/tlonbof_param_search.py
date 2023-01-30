import src.constants as cst

HP_TLONBoF = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.00001},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32, 64, 128]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [15]},
}

HP_TLONBoF_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 15
}

HP_TLONBoF_LOBSTER_FIXED = {

}