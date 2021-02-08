# EqualledArgumentedCycleGAN_Styled_lotated

resume = ''
weight = ''

trial = 3

# Trainig strategy
train = dict(
    batchsize = 20,
    iterations = 80000,
    class_equal = False,
    dataset_dirs = ['data/data/user01/first',
                    'data/data/user02/first',
                    'data/data/user03/first',
                    'data/data/user04/first',
                    'data/data/user08/first',
                    'data/data/user09/first'
                    ],
    out = f'results/gesture/trial{trial}',
    generator = dict(
        model = 'EqualledCycleGAN_Generator7',
        norm = 'batch',
        top = 100,
        use_sigmoid = True,
        ),
    discriminator = dict(
        model = 'EqualledCycleGAN_ArgumentedDiscriminator3',
        dropout = False,
        norm = 'batch',
        top = 100,
        ),
    snapshot_interval = 1000,
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
        lam_g_ges = 10.,
        lam_d_ges = 10., 
        lam_g_user = 1.,
        lam_d_user = 1., 
        lam_g_sm = 1.,
        # lam_d_gp = 0.1,
        # lam_d_drift = 0.1,
        learning_rate_anneal = 0.05,
        learning_rate_anneal_interval = 1000,
        ),
    )

iter = 80000
user = 9

# Testing strategy
test = dict(
    gen = f'results/gesture/trial{trial}/gen_iter_{iter}.npz',
    dataset = f'data/data/user{user+1:02}/first',
    out = f'results/gesture/trial{trial}/iter{iter}/user{user}',
    )
