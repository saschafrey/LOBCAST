
import src.config as co

sweep_configuration_dlb = {

    'parameters': {

        # co.TuningVars.EPOCHS.value: {'values': [5, 10, 15]},
        # co.TuningVars.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},  # 'max': 0.001, 'min': 0.0001
        co.TuningVars.BATCH_SIZE.value: {'values': [32, 68, 128]}, # [32, 64, 128]
        co.TuningVars.IS_SHUFFLE.value: {'values': [True]},

        co.TuningVars.MLP_HIDDEN.value: {'values': [200]},  # [150, 175, 200]

    }
}
