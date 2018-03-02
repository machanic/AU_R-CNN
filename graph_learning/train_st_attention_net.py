import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from graph_learning.dataset.graph_dataset_reader import GlobalDataSet
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from graph_learning.dataset.graph_dataset import GraphDataset
from graph_learning.extensions.AU_roi_label_split_evaluator import ActionUnitEvaluator
from graph_learning.extensions.opencrf_evaluator import OpenCRFEvaluator
from graph_learning.model.st_attention_net.st_attention_net_plus import StAttentioNetPlus
from graph_learning.model.st_attention_net.st_relation_net_plus import StRelationNetPlus
from graph_learning.model.st_attention_net.enum_type import RecurrentType, NeighborMode, SpatialEdgeMode
from graph_learning.updater.bptt_updater import BPTTUpdater
from graph_learning.dataset.crf_pact_structure import CRFPackageStructure
from graph_learning.updater.bptt_updater import convert
from collections import OrderedDict
import os
from graph_learning.trigger.EarlyStopTrigger import EarlyStoppingTrigger
import matplotlib
matplotlib.use('Agg')
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=25,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument('--step_size', '-ss', type=int, default=3000,
                        help='step_size for lr exponential')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot', '-snap', type=int, default=1, help='snapshot epochs for save checkpoint')
    parser.add_argument('--test_mode', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument("--test", '-tt', default='test',help='Test directory path contains test txt file to evaluation')
    parser.add_argument('--train', '-t', default="train",
                        help='Train directory path contains train txt file')
    parser.add_argument('--database',  default="BP4D",
                        help='database to train for')
    parser.add_argument('--lr', '-l', type=float, default=0.01)
    parser.add_argument('--neighbor_mode', type=NeighborMode, choices=list(NeighborMode), help='1:concat_all,2:attention_fuse,3:random_neighbor,4.no_neighbor')
    parser.add_argument('--spatial_edge_mode', type=SpatialEdgeMode, choices=list(SpatialEdgeMode), help='1:all_edge, 2:configure_edge, 3:no_edge')
    parser.add_argument('--temporal_edge_mode',type=RecurrentType, choices=list(RecurrentType), help='1:rnn, 2:attention_block, 3.point-wise feed forward(no temporal)')
    parser.add_argument("--use_relation_net", action='store_true', help='whether to use st_relation_net instead of st_attention_net')
    parser.add_argument("--relation_net_lstm_first", action='store_true',
                        help='whether to use relation_net_lstm_first_forward in st_relation_net')
    parser.add_argument('--use_geometry_features',action='store_true', help='whether to use geometry features')
    parser.add_argument("--num_attrib", type=int, default=2048, help="number of dimension of each node feature")
    parser.add_argument('--geo_num_attrib', type=int, default=4, help='geometry feature length')
    parser.add_argument('--attn_heads', type=int, default=16, help='attention heads number')
    parser.add_argument('--layers', type=int, default=1, help='edge rnn and node rnn layer')
    parser.add_argument("--use_paper_num_label", action="store_true", help="only to use paper reported number of labels"
                                                                           " to train")
    parser.add_argument("--bi_lstm", action="store_true", help="whether to use bi-lstm as Edge/Node RNN")
    parser.add_argument('--weight_decay',type=float,default=0.0005, help="weight decay")
    parser.add_argument("--proc_num",'-proc', type=int,default=1, help="process number of dataset reader")
    parser.add_argument("--resume",action="store_true", help="whether to load npz pretrained file")
    parser.add_argument('--resume_model', '-rm', help='The relative path to restore model file')
    parser.add_argument("--snap_individual", action="store_true", help='whether to snap shot each fixed step into '
                                                                       'individual model file')
    parser.add_argument("--vis", action='store_true', help='whether to visualize computation graph')



    parser.set_defaults(test=False)
    args = parser.parse_args()
    if args.use_relation_net:
        args.out += "_relationnet"
        print("output file to : {}".format(args.out))
    print_interval = 1, 'iteration'
    val_interval = 5, 'iteration'
    print("""
    ======================================
        argument: 
            neighbor_mode:{0}
            spatial_edge_mode:{1}
            temporal_edge_mode:{2}
            use_geometry_features:{3}
    ======================================
    """.format(args.neighbor_mode, args.spatial_edge_mode, args.temporal_edge_mode, args.use_geometry_features))
    adaptive_AU_database(args.database)
    # for the StructuralRNN constuctor need first frame factor graph_backup
    dataset = GlobalDataSet(num_attrib=args.num_attrib, num_geo_attrib=args.geo_num_attrib,
                            train_edge="all")
    file_name = list(filter(lambda e: e.endswith(".txt"), os.listdir(args.train)))[0]

    paper_report_label = OrderedDict()
    if args.use_paper_num_label:
        for AU_idx,AU in sorted(config.AU_SQUEEZE.items(), key=lambda e:int(e[0])):
            if args.database == "BP4D":
                paper_use_AU = config.paper_use_BP4D
            elif args.database =="DISFA":
                paper_use_AU = config.paper_use_DISFA
            if AU in paper_use_AU:
                paper_report_label[AU_idx] = AU
    paper_report_label_idx = list(paper_report_label.keys())
    if not paper_report_label_idx:
        paper_report_label_idx = None


    sample = dataset.load_data(args.train + os.sep + file_name, npy_in_parent_dir=False,
                               paper_use_label_idx=paper_report_label_idx)  # we load first sample for construct S-RNN, it must passed to constructor argument
    crf_pact_structure = CRFPackageStructure(sample, dataset, num_attrib=dataset.num_attrib_type)  # 只读取其中的一个视频的第一帧，由于node个数相对稳定，因此可以construct RNN
    # 因为我们用多分类的hinge loss，所以需要num_label是来自于2进制形式的label数+1（+1代表全0）\

    if args.use_relation_net:
        model = StRelationNetPlus(crf_pact_structure, in_size=dataset.num_attrib_type, out_size=dataset.label_bin_len,
                              database=args.database, neighbor_mode=args.neighbor_mode,
                              spatial_edge_mode=args.spatial_edge_mode, recurrent_block_type=args.temporal_edge_mode,
                              attn_heads=args.attn_heads, dropout=0.5, use_geometry_features=args.use_geometry_features,
                              layers=args.layers, bi_lstm=args.bi_lstm, lstm_first_forward=args.relation_net_lstm_first)
    else:
        model = StAttentioNetPlus(crf_pact_structure, in_size=dataset.num_attrib_type, out_size=dataset.label_bin_len,
                              database=args.database, neighbor_mode=args.neighbor_mode,
                              spatial_edge_mode=args.spatial_edge_mode, recurrent_block_type=args.temporal_edge_mode,
                              attn_heads=args.attn_heads, dropout=0.5, use_geometry_features=args.use_geometry_features,
                              layers=args.layers, bi_lstm=args.bi_lstm)

    # note that the following code attrib_size will be used by open_crf for parameter number, thus we cannot pass dataset.num_attrib_type!
    train_data = GraphDataset(args.train, attrib_size=dataset.num_attrib_type, global_dataset=dataset, need_s_rnn=True,
                              need_cache_factor_graph=False, npy_in_parent_dir=False, get_geometry_feature=True,
                              paper_use_label_idx=paper_report_label_idx)  # train 传入文件夹

    train_iter = chainer.iterators.SerialIterator(train_data, 1, shuffle=True, repeat=True)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        if args.use_relation_net:
            model.st_relation_net.to_gpu(args.gpu)
        else:
            model.st_attention_net.to_gpu(args.gpu)

    specific_key = "all_AU_train"
    if paper_report_label_idx:
        specific_key = "paper_AU_num_train"

    optimizer_snapshot_name = "{0}@{1}@st_attention_network_optimizer@{2}@{3}@{4}@{5}.npz".format(args.database,
                                                                                            specific_key,
                                                                                              args.neighbor_mode,
                                                                                              args.spatial_edge_mode,
                                                                                              args.temporal_edge_mode,
                                                                                              "use_geo" if args.use_geometry_features else "no_geo")
    model_snapshot_name = "{0}@{1}@st_attention_network_model@{2}@{3}@{4}@{5}.npz".format(args.database,
                                                                                          specific_key,
                                                                                      args.neighbor_mode,
                                                                                      args.spatial_edge_mode,
                                                                                      args.temporal_edge_mode,
                                                                                      "use_geo" if args.use_geometry_features else "no_geo")
    if args.snap_individual:
        model_snapshot_name = "{0}@{1}@st_attention_network_model_snapshot_".format(args.database,specific_key)
        model_snapshot_name += "{.updater.iteration}"
        model_snapshot_name += "@{0}@{1}@{2}@{3}.npz".format(args.neighbor_mode,
                                                             args.spatial_edge_mode,
                                                             args.temporal_edge_mode,
                                                             "use_geo" if args.use_geometry_features else "no_geo")
    if os.path.exists(args.out + os.sep + model_snapshot_name):
        print("found trained model file. load trained file: {}".format(args.out + os.sep + model_snapshot_name))
        chainer.serializers.load_npz(args.out + os.sep + model_snapshot_name, model)

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))
    updater = BPTTUpdater(train_iter, optimizer, int(args.gpu))
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    interval = (1, 'iteration')
    if args.test_mode:
        chainer.config.train = False
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=print_interval)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss', "main/accuracy",
         ]), trigger=print_interval)

    log_name = "st_attention_network_{0}@{1}@{2}@{3}@{4}.log".format(args.database,
                                                                      args.neighbor_mode,
                                                                      args.spatial_edge_mode,
                                                                      args.temporal_edge_mode,
                                                                "use_geo" if args.use_geometry_features else "no_geo")

    trainer.extend(chainer.training.extensions.LogReport(trigger=interval,log_name=log_name))
    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1, training_length=(args.epoch, 'epoch')))

    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=optimizer_snapshot_name),
        trigger=(args.snapshot, 'epoch'))

    trainer.extend(
        chainer.training.extensions.snapshot_object(model,
                                                    filename=model_snapshot_name),
        trigger=(args.snapshot, 'epoch'))

    trainer.extend(chainer.training.extensions.ExponentialShift('lr',0.1), trigger=(10, "epoch"))

    if args.resume and os.path.exists(args.out + os.sep + args.resume_model):
        print("loading model_snapshot_name to model")
        chainer.serializers.load_npz(args.out + os.sep + args.resume_model, model)
    if args.resume and os.path.exists(args.out + os.sep + optimizer_snapshot_name):
        print("loading optimizer_snapshot_name to optimizer")
        chainer.serializers.load_npz(args.out + os.sep + optimizer_snapshot_name, optimizer)

    if chainer.training.extensions.PlotReport.available():
        trainer.extend(chainer.training.extensions.PlotReport(['main/loss'],
                                                              file_name="train_loss.png"),
                                                              trigger=val_interval)
        trainer.extend(chainer.training.extensions.PlotReport(['main/accuracy'],
                                                              file_name="train_accuracy.png"), trigger=val_interval)

    trainer.run()

    # cProfile.runctx("trainer.run()", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()

    # # evaluate the final model
    # test_data = S_RNNPlusDataset(args.valid, attrib_size=args.hidden_size, global_dataset=dataset)
    # test_iter = chainer.iterators.SerialIterator(test_data, 1, shuffle=False, repeat=False)
    # gpu = int(args.gpu)
    # chainer.cuda.get_device_from_id(gpu).use()
    # if args.with_crf:
    #     evaluator = OpenCRFEvaluator(iterator=test_iter, target=model, device=-1)
    # else:
    #     evaluator = ActionUnitEvaluator(iterator=validate_iter, target=model, device=-1)
    # result = evaluator()
    # print("final test data result: loss: {0}, accuracy:{1}".format(result["main/loss"], result["main/accuracy"]))

if __name__ == "__main__":
    main()