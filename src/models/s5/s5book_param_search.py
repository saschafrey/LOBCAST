
import src.constants as cst

HP_S5 = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [None]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.01]},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [100]},
    cst.LearningHyperParameter.N_MSG_LAYERS.value: {'values': [2]},
    cst.LearningHyperParameter.N_BOOK_PRE_LAYERS.value: {'values': [1]},
    cst.LearningHyperParameter.N_BOOK_POST_LAYERS.value: {'values': [1]},
    cst.LearningHyperParameter.N_LAYERS.value: {'values': [6]},
    cst.LearningHyperParameter.SSM_LR_BASE.value: {'values': [1e-3]},
    cst.LearningHyperParameter.LR_FACTOR.value: {'values': [1]},
    cst.LearningHyperParameter.CONJ_SYM.value: {'values': [True]},
    cst.LearningHyperParameter.D_MODEL.value: {'values': [32]},
    cst.LearningHyperParameter.SSM_SIZE.value: {'values': [32]},
    cst.LearningHyperParameter.N_BLOCKS.value: {'values': [8]},
    cst.LearningHyperParameter.C_INIT.value: {'values': [cst.C_Initialisers.TRUNC.value]},
    cst.LearningHyperParameter.CLASS_MODE.value: {'values': [cst.ClassificationModes.POOL.value]},
    cst.LearningHyperParameter.ACT_FUNC.value: {'values': [cst.ActivationFunctions.HALFGLU1.value]},
    cst.LearningHyperParameter.PRENORM.value: {'values': [True]},
    cst.LearningHyperParameter.BATCHNORM.value: {'values': [True]},
    cst.LearningHyperParameter.BN_MOMENTUM.value: {'values': [0.95]},
    cst.LearningHyperParameter.OPTIM_CONFIG.value: {'values': [cst.OptimisationConfigurations.STANDARD.value]},
    cst.LearningHyperParameter.DT_GLOBAL.value: {'values': [False]},
    cst.LearningHyperParameter.DT_MIN.value: {'values': [0.001]},
    cst.LearningHyperParameter.DT_MAX.value: {'values': [0.1]},
    cst.LearningHyperParameter.CLIP_EIGS.value: {'values': [False]},
    cst.LearningHyperParameter.BIDIRECTIONAL.value: {'values': [False]},
    cst.LearningHyperParameter.DISCRETIZATION.value : {'values': [cst.DiscretizationMethods.ZOH.value]},

}

HP_S5_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: None,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 100,
    cst.LearningHyperParameter.N_MSG_LAYERS.value: 2,
    cst.LearningHyperParameter.N_BOOK_PRE_LAYERS.value: 1,
    cst.LearningHyperParameter.N_BOOK_POST_LAYERS.value: 1,
    cst.LearningHyperParameter.N_LAYERS.value: 6,
    cst.LearningHyperParameter.SSM_LR_BASE.value: 1e-3,
    cst.LearningHyperParameter.LR_FACTOR.value: 1,
    cst.LearningHyperParameter.CONJ_SYM.value: True,
    cst.LearningHyperParameter.D_MODEL.value: 32,
    cst.LearningHyperParameter.SSM_SIZE.value: 32,
    cst.LearningHyperParameter.N_BLOCKS.value: 8,
    cst.LearningHyperParameter.C_INIT.value: cst.C_Initialisers.TRUNC.value,
    cst.LearningHyperParameter.CLASS_MODE.value: cst.ClassificationModes.POOL.value,
    cst.LearningHyperParameter.ACT_FUNC.value: cst.ActivationFunctions.HALFGLU1.value,
    cst.LearningHyperParameter.PRENORM.value: True,
    cst.LearningHyperParameter.BATCHNORM.value: True,
    cst.LearningHyperParameter.BN_MOMENTUM.value:0.95,
    cst.LearningHyperParameter.OPTIM_CONFIG.value: cst.OptimisationConfigurations.STANDARD.value,
    cst.LearningHyperParameter.DT_GLOBAL.value: False,
    cst.LearningHyperParameter.DT_MIN.value: 0.001,
    cst.LearningHyperParameter.DT_MAX.value: 0.1,
    cst.LearningHyperParameter.CLIP_EIGS.value: False,
    cst.LearningHyperParameter.BIDIRECTIONAL.value: False,
    cst.LearningHyperParameter.DISCRETIZATION.value : cst.DiscretizationMethods.ZOH.value,
}


HP_S5_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: None,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 100,
    cst.LearningHyperParameter.N_MSG_LAYERS.value: 2,
    cst.LearningHyperParameter.N_BOOK_PRE_LAYERS.value: 1,
    cst.LearningHyperParameter.N_BOOK_POST_LAYERS.value: 1,
    cst.LearningHyperParameter.N_LAYERS.value: 6,
    cst.LearningHyperParameter.SSM_LR_BASE.value: 1e-3,
    cst.LearningHyperParameter.LR_FACTOR.value: 1,
    cst.LearningHyperParameter.CONJ_SYM.value: True,
    cst.LearningHyperParameter.D_MODEL.value: 32,
    cst.LearningHyperParameter.SSM_SIZE.value: 32,
    cst.LearningHyperParameter.N_BLOCKS.value: 8,
    cst.LearningHyperParameter.C_INIT.value: cst.C_Initialisers.TRUNC.value,
    cst.LearningHyperParameter.CLASS_MODE.value: cst.ClassificationModes.POOL.value,
    cst.LearningHyperParameter.ACT_FUNC.value: cst.ActivationFunctions.HALFGLU1.value,
    cst.LearningHyperParameter.PRENORM.value: True,
    cst.LearningHyperParameter.BATCHNORM.value: True,
    cst.LearningHyperParameter.BN_MOMENTUM.value:0.95,
    cst.LearningHyperParameter.OPTIM_CONFIG.value: cst.OptimisationConfigurations.STANDARD.value,
    cst.LearningHyperParameter.DT_GLOBAL.value: False,
    cst.LearningHyperParameter.DT_MIN.value: 0.001,
    cst.LearningHyperParameter.DT_MAX.value: 0.1,
    cst.LearningHyperParameter.CLIP_EIGS.value: False,
    cst.LearningHyperParameter.BIDIRECTIONAL.value: False,
    cst.LearningHyperParameter.DISCRETIZATION.value : cst.DiscretizationMethods.ZOH.value,
}