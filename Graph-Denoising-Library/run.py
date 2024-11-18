import argparse
import os
import torch
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_imputation import Exp_Imputation
# from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
# from exp.exp_anomaly_detection import Exp_Anomaly_Detection
# from exp.exp_classification import Exp_Classification
# from utils.print_args import print_args
import random
import numpy as np
from exp.exp_classification import Exp_Classification
from exp.exp_community_detection import Exp_Community_Detection


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GraphDenoising')
    # model name
    parser.add_argument('--model', type=str, default='DropEdge',
                        help='model name, options: [DropEdge]')

    
    
    # Training parameter  required=true表示必传参数
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    # todo 需要搞清楚下列训练参数意义
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Disable validation during training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--lradjust', action='store_true',
                        default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--mixmode", action="store_true",
                        default=False, help="Enable CPU GPU mixing mode.")
    parser.add_argument("--warm_start", default="",
                        help="The model name to be loaded for warm start.")
    parser.add_argument('--debug', action='store_true',
                        default=False, help="Enable the detialed training output.")
    parser.add_argument('--dataset', default="cora", help="The data set")
    parser.add_argument('--datapath', default="data/cora", help="The data path.")
    parser.add_argument("--early_stopping", type=int,
                        default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
    parser.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")


    parser.add_argument('--is_training', type=int, default=1, help='status')
    # Model parameter 针对不同模型 有不同的模型参数 尽量设定一个默认值
    # 尽量是双引号
    parser.add_argument('--model_type', default="mutigcn",
                        help="Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)")
    parser.add_argument('--inputlayer', type=str, default='gcn',
                        help="The input layer of the model.")
    parser.add_argument('--outputlayer', default='gcn',
                        help="The output layer of the model.")
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--withbn', action='store_true', default=False,
                        help='Enable Bath Norm GCN')
    parser.add_argument('--withloop', action="store_true", default=False,
                        help="Enable loop layer GCN")
    parser.add_argument('--nhiddenlayer', type=int, default=1,
                        help='The number of hidden layers.')
    parser.add_argument("--normalization", default="AugNormAdj",
                        help="The normalization on the adj matrix.")
    parser.add_argument("--sampling_percent", type=float, default=1.0,
                        help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
    # parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
    parser.add_argument("--nbaseblocklayer", type=int, default=1,
                        help="The number of layers in each baseblock")
    parser.add_argument("--aggrmethod", default="default",
                        help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn")
    parser.add_argument("--task_type", default="full", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

    parser.add_argument("--task_name", default="classification")

    


    # 实验次数
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    args = parser.parse_args()
    # pre setting
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.mixmode = args.no_cuda and args.mixmode and torch.cuda.is_available()
    # 以下属于dropedge的特定操作 应该转移到dropedge中
    if args.aggrmethod == "default":
        if args.model_type == "resgcn":
            args.aggrmethod = "add"
        else:
            args.aggrmethod = "concat"
    if args.fastmode and args.early_stopping > 0:
        args.early_stopping = 0
        print("In the fast mode, early_stopping is not valid option. Setting early_stopping = 0.")
    if args.model_type == "mutigcn":
        print("For the multi-layer gcn model, the aggrmethod is fixed to nores and nhiddenlayers = 1.")
        args.nhiddenlayer = 1
        args.aggrmethod = "nores"

    # random seed setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda or args.mixmode:
        torch.cuda.manual_seed(args.seed)
        
    # 判断任务种类 选定实现类 
    if args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'communityDetection':
        Exp = Exp_Community_Detection

    # 传入参数进行模型的实现和训练测试(1表示需要训练  2直接测试)
    if args.is_training:
        for ii in range(args.itr): # 根据实验次数进行实验
            # setting record of experiments
            exp = Exp(args)  # set experiments 传入命令行参数

            exp.train()
            exp.test()
            torch.cuda.empty_cache()  



# if __name__ == '__main__':
#     fix_seed = 2021
#     random.seed(fix_seed)
#     torch.manual_seed(fix_seed)
#     np.random.seed(fix_seed)

#     parser = argparse.ArgumentParser(description='TimesNet')

#     # basic config
#     parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
#                         help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
#     parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
#     parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
#     parser.add_argument('--model', type=str, required=True, default='Autoformer',
#                         help='model name, options: [Autoformer, Transformer, TimesNet]')

#     # data loader
#     parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
#     parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
#     parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
#     parser.add_argument('--features', type=str, default='M',
#                         help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#     parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
#     parser.add_argument('--freq', type=str, default='h',
#                         help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

#     # forecasting task
#     parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=48, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
#     parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
#     parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

#     # inputation task
#     parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

#     # anomaly detection task
#     parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

#     # model define
#     parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
#     parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
#     parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
#     parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
#     parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
#     parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=7, help='output size')
#     parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
#     parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
#     parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
#     parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
#     parser.add_argument('--factor', type=int, default=1, help='attn factor')
#     parser.add_argument('--distil', action='store_false',
#                         help='whether to use distilling in encoder, using this argument means not using distilling',
#                         default=True)
#     parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
#     parser.add_argument('--embed', type=str, default='timeF',
#                         help='time features encoding, options:[timeF, fixed, learned]')
#     parser.add_argument('--activation', type=str, default='gelu', help='activation')
#     parser.add_argument('--channel_independence', type=int, default=1,
#                         help='0: channel dependence 1: channel independence for FreTS model')
#     parser.add_argument('--decomp_method', type=str, default='moving_avg',
#                         help='method of series decompsition, only support moving_avg or dft_decomp')
#     parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
#     parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
#     parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
#     parser.add_argument('--down_sampling_method', type=str, default=None,
#                         help='down sampling method, only support avg, max, conv')
#     parser.add_argument('--seg_len', type=int, default=48,
#                         help='the length of segmen-wise iteration of SegRNN')

#     # optimization
#     parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=1, help='experiments times')
#     parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='test', help='exp description')
#     parser.add_argument('--loss', type=str, default='MSE', help='loss function')
#     parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
#     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

#     # GPU
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

#     # de-stationary projector params
#     parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
#                         help='hidden layer dimensions of projector (List)')
#     parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

#     # metrics (dtw)
#     parser.add_argument('--use_dtw', type=bool, default=False, 
#                         help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
#     # Augmentation
#     parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
#     parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
#     parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
#     parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
#     parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
#     parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
#     parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
#     parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
#     parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
#     parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
#     parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
#     parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
#     parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
#     parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
#     parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
#     parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
#     parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
#     parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")


#     # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
#     args.use_gpu = True if torch.cuda.is_available() else False

#     print(torch.cuda.is_available())

#     if args.use_gpu and args.use_multi_gpu:
#         args.devices = args.devices.replace(' ', '')
#         device_ids = args.devices.split(',')
#         args.device_ids = [int(id_) for id_ in device_ids]
#         args.gpu = args.device_ids[0]

#     print('Args in experiment:')
#     print_args(args)

#     if args.task_name == 'long_term_forecast':
#         Exp = Exp_Long_Term_Forecast
#     elif args.task_name == 'short_term_forecast':
#         Exp = Exp_Short_Term_Forecast
#     elif args.task_name == 'imputation':
#         Exp = Exp_Imputation
#     elif args.task_name == 'anomaly_detection':
#         Exp = Exp_Anomaly_Detection
#     elif args.task_name == 'classification':
#         Exp = Exp_Classification
#     else:
#         Exp = Exp_Long_Term_Forecast

#     if args.is_training:
#         for ii in range(args.itr):
#             # setting record of experiments
#             exp = Exp(args)  # set experiments
#             setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
#                 args.task_name,
#                 args.model_id,
#                 args.model,
#                 args.data,
#                 args.features,
#                 args.seq_len,
#                 args.label_len,
#                 args.pred_len,
#                 args.d_model,
#                 args.n_heads,
#                 args.e_layers,
#                 args.d_layers,
#                 args.d_ff,
#                 args.expand,
#                 args.d_conv,
#                 args.factor,
#                 args.embed,
#                 args.distil,
#                 args.des, ii)

#             print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#             exp.train(setting)

#             print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#             exp.test(setting)
#             torch.cuda.empty_cache()
#     else:
#         ii = 0
#         setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
#             args.task_name,
#             args.model_id,
#             args.model,
#             args.data,
#             args.features,
#             args.seq_len,
#             args.label_len,
#             args.pred_len,
#             args.d_model,
#             args.n_heads,
#             args.e_layers,
#             args.d_layers,
#             args.d_ff,
#             args.expand,
#             args.d_conv,
#             args.factor,
#             args.embed,
#             args.distil,
#             args.des, ii)

#         exp = Exp(args)  # set experiments
#         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#         exp.test(setting, test=1)
#         torch.cuda.empty_cache()
