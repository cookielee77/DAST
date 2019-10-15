import os
import sys
import pprint
import time
import argparse
import logging
from pathlib import Path

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    # data path
    argparser.add_argument('--dataDir',
            type=str,
            default='')
    argparser.add_argument('--dataset',
            type=str,
            default='',
            help='if doman_adapt enable, dataset means target dataset')
    argparser.add_argument('--modelDir',
            type=str,
            default='')
    argparser.add_argument('--logDir',
            type=str,
            default='')

    # general model setting
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.0005)
    argparser.add_argument('--batch_size',
            type=int,
            default=64)
    argparser.add_argument('--pretrain_epochs',
            type=int,
            default=10,
            help='max pretrain epoch for LM.')
    argparser.add_argument('--max_epochs',
            type=int,
            default=20)
    argparser.add_argument('--max_len',
            type=int,
            default=20,
            help='the max length of sequence')
    argparser.add_argument('--noise_word',
            action='store_true',
            help='whether add noise in enc batch.')
    argparser.add_argument('--trim_padding',
            action='store_true',
            help='whether trim the padding in each batch.')
    argparser.add_argument('--order_data',
            action='store_true',
            help='whether order the data according the length in the dataset.')

    # CNN model
    argparser.add_argument('--filter_sizes',
            type=str,
            default='1,2,3,4,5')
    argparser.add_argument('--n_filters',
            type=int,
            default=128)
    argparser.add_argument('--confidence',
            type=float,
            default=0.8,
            help='The classification confidence used to filter the data')

    # style transfer model
    argparser.add_argument('--network',
            type=str,
            default='',
            help='The style transfer network path')
    argparser.add_argument('--rho',                 # loss_rec + rho * loss_adv
            type=float,
            default=1)
    argparser.add_argument('--gamma_init',          # softmax(logit / gamma)
            type=float,
            default=0.1)
    argparser.add_argument('--gamma_decay',
            type=float,
            default=1)
    argparser.add_argument('--gamma_min',
            type=float,
            default=0.1)
    argparser.add_argument('--beam',
            type=int,
            default=1)
    argparser.add_argument('--dropout_rate',
            type=float,
            default=0.5)
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_y',
            type=int,
            default=200)
    argparser.add_argument('--dim_z',
            type=int,
            default=500)
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)

    # training config
    argparser.add_argument('--suffix',
            type=str,
            default='')
    argparser.add_argument('--load_model',
            action='store_true',
            help='whether load the model for test')
    argparser.add_argument('--save_model',
            action='store_true',
            help='whether save the model for test')
    argparser.add_argument('--train_checkpoint_frequency',
            type=int,
            default=4,
            help='how many checkpoints in one training epoch')
    argparser.add_argument('--training_portion',
            type=float,
            default=1.0)
    argparser.add_argument('--source_training_portion',
            type=float,
            default=1.0)

    # Multi-dataset support
    argparser.add_argument('--domain_adapt',
            action='store_true',
            help='whether use multidataset for domain-adaptation')
    argparser.add_argument('--source_dataset',
            type=str,
            default='yelp')
    argparser.add_argument('--dim_d',
            type=int,
            default=50,
            help='The dimension of domain vector.')
    argparser.add_argument('--alpha',
            type=float,
            default=0.0,
            help='The weight of domain loss.')

    # Yelp/Amazon online dataset for test only
    argparser.add_argument('--online_test',
            action='store_true',
            help='whether to use human annotated sentences to evalute the bleu.')
    argparser.add_argument('--save_samples',
            action='store_true',
            help='whether to save validation samples from the model.')


    args = argparser.parse_args()
    # check whether use online annotated dataset from human
    if args.dataset in ['yelp', 'amazon']:
        args.online_test = True

    # update data path according to single dataset or multiple dataset
    if args.domain_adapt:
        args = update_domain_adapt_datapath(args)
    else:
        args.dataDir = os.path.join(args.dataDir, 'data')
        data_root = os.path.join(args.dataDir, args.dataset)
        args.train_path = os.path.join(data_root, 'train')
        args.valid_path = os.path.join(data_root, 'valid')
        args.test_path = os.path.join(data_root, 'test')
        args.vocab = os.path.join(data_root, 'vocab')

        # update output path
        args.modelDir = os.path.join(args.modelDir, 'save_model')
        args.classifier_path = os.path.join(args.modelDir, 'classifier', args.dataset)
        args.lm_path = os.path.join(args.modelDir, 'lm', args.dataset)
        args.styler_path = os.path.join(args.modelDir, 'styler')

    # update batch size if using parallel training
    if 'para' in args.dataset:
        args.batch_size = int(args.batch_size/2)

    # update output path
    if not args.logDir:
        # if not in philly enviroment
        args.logDir = 'logs'
        args.logDir = os.path.join(args.logDir, args.network, args.suffix)
    log_dir = Path(args.logDir)

    if not log_dir.exists():
        print('=> creating {}'.format(log_dir))
        log_dir.mkdir(parents = True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')        
    log_file = '{}_{}_{}.log'.format(args.network, args.suffix, time_str)
    # update the suffix for tensorboard file name
    args.suffix = '{}_{}_{}'.format(args.network, args.suffix, time_str)

    final_log_file = log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)


    logger.info('------------------------------------------------')
    logger.info(pprint.pformat(args))
    logger.info('------------------------------------------------')

    return args

def update_domain_adapt_datapath(args):
    # update data path
    args.dataDir = os.path.join(args.dataDir, 'data')
    # target_data
    target_data_root = os.path.join(args.dataDir, args.dataset)
    args.target_train_path = os.path.join(target_data_root, 'train')
    args.target_valid_path = os.path.join(target_data_root, 'valid')
    args.target_test_path = os.path.join(target_data_root, 'test')
    # the vocabulary used for classifier evaluation
    args.target_vocab = os.path.join(target_data_root, 'vocab')

    # source data
    source_data_root = os.path.join(args.dataDir, args.source_dataset)
    args.source_train_path = os.path.join(source_data_root, 'train')
    args.source_valid_path = os.path.join(source_data_root, 'valid')
    args.source_test_path = os.path.join(source_data_root, 'test')
    # the vocabulary used for classifier evaluation
    args.source_vocab = os.path.join(source_data_root, 'vocab')

    # save the togather vocab in common root 'data/multi_vocab'
    args.multi_vocab = os.path.join(
        args.dataDir, '_'.join([args.source_dataset, args.dataset, 'multi_vocab']))

    # update output path
    args.modelDir = os.path.join(args.modelDir, 'save_model')
    args.target_classifier_path = os.path.join(args.modelDir, 'classifier', args.dataset)
    args.source_classifier_path = os.path.join(args.modelDir, 'classifier', args.source_dataset)
    args.domain_classifier_path = os.path.join(
        args.modelDir, 'classifier', '_'.join([args.source_dataset, args.dataset, 'domain_adapt']))
    args.styler_path = os.path.join(args.modelDir, 'domain_adapt_styler')

    return args

