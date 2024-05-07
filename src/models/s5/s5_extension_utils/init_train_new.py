from LOBS5Prediction.lob.init_train import *
from src.config import Configuration
import src.constants as cst

def init_train_state(
        config:Configuration,
        in_dim:int,
        n_classes: int,
        seq_len: int,
        book_dim: int,
        book_seq_len,
        print_shapes=False
    ) -> Tuple[TrainState, Union[partial[BatchLobPredModel], partial[FullLobPredModel],partial[BatchBookOnlyPredModel]]]:


    ssm_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SSM_SIZE]
    ssm_lr = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SSM_LR_BASE]

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = config.HYPER_PARAMETERS[cst.LearningHyperParameter.LR_FACTOR] * ssm_lr

    # determine the size of initial blocks
    n_blocks=config.HYPER_PARAMETERS[cst.LearningHyperParameter.N_BLOCKS]
    block_size = int(ssm_size / n_blocks)

    key = random.PRNGKey(config.SEED)
    init_rng, train_rng = random.split(key, num=2)

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONJ_SYM]:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((n_blocks, block_size))).ravel()
    V = block_diag(*([V] * n_blocks))
    Vinv = block_diag(*([Vc] * n_blocks))

    if print_shapes:
        print("Lambda.shape={}".format(Lambda.shape))
        print("V.shape={}".format(V.shape))
        print("Vinv.shape={}".format(Vinv.shape))
        print("book_seq_len", book_seq_len)
        print("book_dim", book_dim)

    padded = False
    retrieval = False
    speech = False

    ssm_init_fn = init_S5SSM(
        H=config.HYPER_PARAMETERS[cst.LearningHyperParameter.D_MODEL],
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=config.HYPER_PARAMETERS[cst.LearningHyperParameter.C_INIT],
        discretization=config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRETIZATION],
        dt_min=config.HYPER_PARAMETERS[cst.LearningHyperParameter.DT_MIN],
        dt_max=config.HYPER_PARAMETERS[cst.LearningHyperParameter.DT_MAX],
        conj_sym=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONJ_SYM],
        clip_eigs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CLIP_EIGS],
        bidirectional=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BIDIRECTIONAL]
    )
    if config.CHOSEN_MODEL==cst.Models.S5MSGSBOOK:
        # if args.num_devices > 1:
        #     model_cls = ParFullLobPredModel
        # else:
        #     model_cls = BatchFullLobPredModel
        
        model_cls = partial(
            # projecting sequence lengths down has appeared better than padding
            BatchFullLobPredModel,
            #BatchPaddedLobPredModel,
            #model_cls,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=config.HYPER_PARAMETERS[cst.LearningHyperParameter.D_MODEL],
            d_book=book_dim,
            n_message_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.N_MSG_LAYERS],  # 2
            n_fused_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.N_LAYERS],
            n_book_pre_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.N_BOOK_PRE_LAYERS],
            n_book_post_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.N_BOOK_POST_LAYERS],
            activation=config.HYPER_PARAMETERS[cst.LearningHyperParameter.ACT_FUNC],
            dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT],
            mode=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CLASS_MODE],
            prenorm=config.HYPER_PARAMETERS[cst.LearningHyperParameter.PRENORM],
            batchnorm=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCHNORM],
            bn_momentum=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BN_MOMENTUM],
        )
    elif config.CHOSEN_MODEL==cst.Models.S5BOOK:
        model_cls = partial(
            BatchBookOnlyPredModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=config.HYPER_PARAMETERS[cst.LearningHyperParameter.D_MODEL],
            d_book=in_dim,
            n_book_pre_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.N_BOOK_PRE_LAYERS],
            n_book_post_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.N_BOOK_POST_LAYERS],
            n_fused_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.N_LAYERS],
            activation=config.HYPER_PARAMETERS[cst.LearningHyperParameter.ACT_FUNC],
            dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT],
            mode=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CLASS_MODE],
            prenorm=config.HYPER_PARAMETERS[cst.LearningHyperParameter.PRENORM],
            batchnorm=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCHNORM],
            bn_momentum=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BN_MOMENTUM],
        )
    elif config.CHOSEN_MODEL==cst.Models.S5MSGS:
        if cst.NUM_GPUS > 1:
            raise NotImplementedError("Message only model not implemented for multi-device training")
    
        model_cls = partial(
            BatchLobPredModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=config.HYPER_PARAMETERS[cst.LearningHyperParameter.D_MODEL],
            n_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.N_LAYERS],
            padded=padded,
            activation=config.HYPER_PARAMETERS[cst.LearningHyperParameter.ACT_FUNC],
            dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT],
            mode=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CLASS_MODE],
            prenorm=config.HYPER_PARAMETERS[cst.LearningHyperParameter.PRENORM],
            batchnorm=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCHNORM],
            bn_momentum=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BN_MOMENTUM],
        )
    use_book_data=False
    if config.CHOSEN_MODEL==cst.Models.S5MSGSBOOK:
        use_book_data=True

    # initialize training state
    state = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        use_book_data=use_book_data,
        in_dim=in_dim,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        bsz=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
        seq_len=seq_len,
        weight_decay=config.HYPER_PARAMETERS[cst.LearningHyperParameter.WEIGHT_DECAY],
        batchnorm=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCHNORM],
        opt_config=config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIM_CONFIG],
        ssm_lr=ssm_lr,
        lr=lr,
        dt_global=config.HYPER_PARAMETERS[cst.LearningHyperParameter.DT_GLOBAL],
        num_devices=cst.NUM_GPUS,
    )

    return state, model_cls