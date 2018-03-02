import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from graph_learning.dataset.graph_dataset_reader import GlobalDataSet
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from graph_learning.dataset.graph_dataset import GraphDataset
from graph_learning.model.graph_attention_networks.graph_attention_rnn import GraphAttentionModel
from graph_learning.updater.bptt_updater import convert,BPTTUpdater
from chainer.training import StandardUpdater
import os
import matplotlib
import config
matplotlib.use('Agg')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument('--step_size', '-ss', type=int, default=3000,
                        help='step_size for lr exponential')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot', '-snap', type=int, default=20, help='snapshot epochs for save checkpoint')
    parser.add_argument('--train', '-t', default="train",
                        help='Train directory path contains train txt file')
    parser.add_argument('--database',  default="BP4D",
                        help='database to train for')
    parser.add_argument('--lr', '-l', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=1024,
                        help="the hidden dimension of the middle layers")
    parser.add_argument("--num_attrib", type=int, default=2048, help="number of dimension of each node feature")
    parser.add_argument("--proc_num",'-proc', type=int,default=1, help="process number of dataset reader")
    parser.add_argument("--need_cache_graph", "-ng", action="store_true",
                        help="whether to cache factor graph to LRU cache")
    parser.add_argument("--resume",action="store_true", help="whether to load npz pretrained file")
    parser.add_argument('--atten_heads',type=int, default=4, help="atten heads for parallel learning")
    parser.add_argument('--layer_num', type=int, default=2, help='layer number of GAT')

    args = parser.parse_args()
    print_interval = 1, 'iteration'
    val_interval = 5, 'iteration'

    adaptive_AU_database(args.database)

    box_num = config.BOX_NUM[args.database]
    # for the StructuralRNN constuctor need first frame factor graph_backup
    dataset = GlobalDataSet(num_attrib=args.num_attrib, train_edge="all")
    file_name = list(filter(lambda e: e.endswith(".txt"), os.listdir(args.train)))[0]
    dataset.load_data(args.train + os.sep + file_name, False)  # we load first sample for construct S-RNN, it must passed to constructor argument
    model = GraphAttentionModel(input_dim=dataset.num_attrib_type, hidden_dim=args.hidden_size, class_number=dataset.label_bin_len,
                                atten_heads=args.atten_heads, layers_num=args.layer_num, frame_node_num=box_num)
    # note that the following code attrib_size will be used by open_crf for parameter number, thus we cannot pass dataset.num_attrib_type!
    train_data = GraphDataset(args.train, attrib_size=2048, global_dataset=dataset, need_s_rnn=False,
                              need_cache_factor_graph=args.need_cache_graph, need_adjacency_matrix=True,
                              npy_in_parent_dir=False, need_factor_graph=False)  # train 传入文件夹
    train_iter = chainer.iterators.SerialIterator(train_data, 1, shuffle=True, repeat=True)
    if args.gpu >= 0:
        print("using gpu : {}".format(args.gpu))
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    updater = BPTTUpdater(train_iter, optimizer,  device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    interval = (1, 'iteration')
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=print_interval)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss', "main/accuracy",
         ]), trigger=print_interval)

    log_name = "GAT.log"
    trainer.extend(chainer.training.extensions.LogReport(trigger=interval,log_name=log_name))
    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1, training_length=(args.epoch, 'epoch')))
    optimizer_snapshot_name = "{0}_GAT_optimizer.npz".format(args.database)
    model_snapshot_name = "{0}_GAT_model.npz".format(args.database)
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=optimizer_snapshot_name),
        trigger=(args.snapshot, 'epoch'))

    trainer.extend(
        chainer.training.extensions.snapshot_object(model,
                                                    filename=model_snapshot_name),
        trigger=(args.snapshot, 'epoch'))
    trainer.extend(chainer.training.extensions.ExponentialShift('lr',0.7), trigger=(5, "epoch"))

    if args.resume and os.path.exists(args.out + os.sep + model_snapshot_name):
        print("loading model_snapshot_name to model")
        chainer.serializers.load_npz(args.out + os.sep + model_snapshot_name, model)
    if args.resume and os.path.exists(args.out + os.sep + optimizer_snapshot_name):
        print("loading optimizer_snapshot_name to optimizer")
        chainer.serializers.load_npz( args.out + os.sep + optimizer_snapshot_name, optimizer)


    if chainer.training.extensions.PlotReport.available():
        trainer.extend(chainer.training.extensions.PlotReport(['main/loss'],
                                                              file_name="train_loss.png"),
                                                              trigger=(100,"iteration"))
        trainer.extend(chainer.training.extensions.PlotReport(['opencrf_val/F1','opencrf_val/accuracy'],
                                                              file_name="{}_val_f1.png"), trigger=val_interval)

    trainer.run()

if __name__ == "__main__":
    main()