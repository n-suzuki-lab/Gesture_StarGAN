# EqualledArgumentedCycleGAN_Styled_lotated

resume = ''
weight = ''

style = 'user' # 'gesture' or 'user'
trial = 1

# Trainig strategy
train = dict(
    batchsize = 20,
    iterations = 50,
    class_equal = False,
    dataset_dirs = ['data/user01/first',
                    'data/user02/first',
                    'data/user03/first',
                    'data/user04/first',
                    'data/user08/first',
                    'data/user09/first'
                    ],
    n_gesture = 4,
    # skip_ges is not loaded for train
    #    style: gesture -> None
    #    style: user -> gesture_id not included in training data
    skip_ges = None, 
    out = f'results/{style}/trial{trial}',
    generator = dict(
        model = 'StarGAN_Generator',
        norm = 'batch',
        top = 100,
        use_sigmoid = True,
        ),
    discriminator = dict(
        model = 'StarGAN_Discriminator',
        dropout = False,
        norm = 'batch',
        top = 100,
        ),
    display_interval = 24,
    preview_interval = 500,
    save_interval = 1000,

    loss_type = 'ls',

    parameters=dict(
        g_lr = 0.0001,
        d_lr = 0.00005,
        lam_g_ad = 1,
        lam_d_ad = 1,
        lam_g_eq = 100.,
        lam_g_rec = 100.,
        lam_g_style = 10.,
        lam_d_style = 10., 
        lam_g_cont = 1.,
        lam_d_cont = 1., 
        lam_g_sm = 1.,
        # lam_d_gp = 0.1,
        # lam_d_drift = 0.1,
        learning_rate_anneal = 0.05,
        learning_rate_anneal_interval = 1000,
        ),
    )

# Testing strategy
gen_model_iter = 80000
source_user = 0

test = dict(
    # gen_path = f'results/{style}/trial{trial}/gen_iter_{gen_model_iter}.npz',
    gen_path = f'results/{style}/trial{trial}/gen.npz',
    source_data = f'data/user{source_user+1:02}/first',
    ges_class = 0,
    target_style = 1,
    out = f'results/{style}/trial{trial}/generated_data/iter{gen_model_iter}',
    )
