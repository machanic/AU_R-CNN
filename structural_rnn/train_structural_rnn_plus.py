import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from structural_rnn.dataset.graph_dataset_reader import GlobalDataSet
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from structural_rnn.dataset.structural_RNN_dataset import S_RNNPlusDataset
from structural_rnn.extensions.AU_evaluator import ActionUnitEvaluator
from structural_rnn.extensions.opencrf_evaluator import OpenCRFEvaluator
from structural_rnn.model.s_rnn.s_rnn_plus import StructuralRNNPlus
from structural_rnn.updater.bptt_updater import BPTTUpdater
from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
from structural_rnn.updater.bptt_updater import convert
import os
from structural_rnn.trigger.EarlyStopTrigger import EarlyStoppingTrigger
from chainer.training.extensions.evaluator import Evaluator
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument('--step_size', '-ss', type=int, default=3000,
                        help='step_size for lr exponential')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot', '-snap', type=float, default=0.5, help='snapshot epochs for save checkpoint')
    parser.add_argument('--test_mode', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--valid', '-v', default='valid',
                        help='Test directory path contains test txt file')
    parser.add_argument("--test", '-tt', default='test',help='Test directory path contains test txt file to evaluation')
    parser.add_argument('--train', '-t', default="train",
                        help='Train directory path contains train txt file')
    parser.add_argument('--database',  default="BP4D",
                        help='database to train for')
    parser.add_argument("--stop_eps", '-eps', type=float, default=1e-3, help="f - old_f < eps ==> early stop")
    parser.add_argument('--with_crf', '-crf', action='store_true', help='whether to use open crf layer')
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--crf_lr', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=512, help="if you want to use open-crf layer, this hidden_size is node dimension input of open-crf")
    parser.add_argument('--eval_mode', action='store_true', help='whether to evaluate the model')
    parser.add_argument("--fold", "-fd", type=int,
                        help="which fold of K-fold")
    parser.add_argument("--split_idx", '-sp', type=int, help="which split_idx")
    parser.add_argument("--proc_num",'-proc', type=int,default=1, help="process number of dataset reader")
    parser.set_defaults(test=False)
    args = parser.parse_args()
    print_interval = 1, 'iteration'
    val_interval = 1, 'iteration'

    adaptive_AU_database(args.database)


    # for the StructuralRNN constuctor need first frame factor graph_backup
    dataset = GlobalDataSet(info_dict_path=os.path.dirname(args.train)+os.sep + "data_info.json")
    file_name = list(filter(lambda e: e.endswith(".txt"), os.listdir(args.train)))[0]
    sample = dataset.load_data(args.train + os.sep + file_name)  # we load first sample for construct S-RNN, it must passed to constructor argument
    crf_pact_structure = CRFPackageStructure(sample, dataset, num_attrib=args.hidden_size)  # 只读取其中的一个视频的第一帧，由于node个数相对稳定，因此可以construct RNN

    print("in_size:{}".format(dataset.num_attrib_type))
    model = StructuralRNNPlus(crf_pact_structure, in_size=dataset.num_attrib_type, out_size=dataset.label_bin_len,
                              hidden_size=args.hidden_size, with_crf=args.with_crf)

    if args.eval_mode:
        test_data = S_RNNPlusDataset(args.test,  attrib_size=args.hidden_size, global_dataset=dataset, need_s_rnn=True)
        test_iter = chainer.iterators.SerialIterator(test_data,1,shuffle=False,repeat=True)
        gpu = int(args.gpu)
        evaluator = ActionUnitEvaluator(test_iter, model, device=gpu,database=args.database)
        observation = evaluator.evaluate()
        with open(args.out + os.sep + "S-eval_result.txt", "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()
        return

    train_data = S_RNNPlusDataset(args.train,  attrib_size=args.hidden_size, global_dataset=dataset, need_s_rnn=True)  # train 传入文件夹
    valid_data = S_RNNPlusDataset(args.valid,  attrib_size=args.hidden_size, global_dataset=dataset, need_s_rnn=True)  # attrib_size控制open-crf层的weight长度

    if args.proc_num == 1:
        train_iter = chainer.iterators.SerialIterator(train_data, 1, shuffle=True, repeat=True)
    elif args.proc_num > 1:
        train_iter = chainer.iterators.MultiprocessIterator(train_data, batch_size=1, n_processes=args.proc_num,
                                                            repeat=True, shuffle=True, n_prefetch=10,
                                                            shared_mem=31457280)
    validate_iter = chainer.iterators.SerialIterator(valid_data, 1, shuffle=False, repeat=False)



    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.structural_rnn.to_gpu(args.gpu)


    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    updater = BPTTUpdater(train_iter, optimizer, int(args.gpu))
    early_stop = EarlyStoppingTrigger(args.epoch, key='main/loss', eps=float(args.stop_eps))
    if args.with_crf:
        trainer = chainer.training.Trainer(updater, stop_trigger=early_stop, out=args.out)
        model.open_crf.W.update_rule.hyperparam.lr = float(args.crf_lr)
        model.open_crf.to_cpu()
    else:
        trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    interval = (1, 'iteration')
    if args.test_mode:
        chainer.config.train = False
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=print_interval)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss', "opencrf_val/main/hit",  # "opencrf_validation/main/U_hit",
         "opencrf_val/main/miss",  # "opencrf_validation/main/U_miss",
         "opencrf_val/main/F1",  # "opencrf_validation/main/U_F1"
         'opencrf_val/main/accuracy',
         ]), trigger=print_interval)

    trainer.extend(chainer.training.extensions.LogReport(trigger=interval,log_name='s_rnn_plus.log'))
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1, training_length=(args.epoch, 'epoch')))
    optimizer_snapshot_name = "{0}_{1}_{2}_srnn_plus_optimizer.npz".format(args.database, args.fold, args.split_idx)
    model_snapshot_name = "{0}_{1}_{2}_srnn_plus_model.npz".format(args.database, args.fold, args.split_idx)
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=optimizer_snapshot_name),
        trigger=(args.snapshot, 'iteration'))

    trainer.extend(
        chainer.training.extensions.snapshot_object(model,
                                                    filename=model_snapshot_name),
        trigger=(args.snapshot, 'iteration'))
    trainer.extend(chainer.training.extensions.ExponentialShift('lr',0.7), trigger=(5, "epoch"))


    if chainer.training.extensions.PlotReport.available():
        # trainer.extend(chainer.training.extensions.PlotReport(['au_validation/main/f1_frame'],file_name='au_validation.png'),
        #                trigger=val_interval)
        trainer.extend(
            chainer.training.extensions.PlotReport(['opencrf_val/main/F1'], file_name='srnn_plus_f1.png'),
            trigger=val_interval)


    # au_evaluator = ActionUnitEvaluator(iterator=validate_iter, target=model, device=-1, database=args.database,
    #                                    data_info_path=os.path.dirname(args.train) + os.sep + "data_info.json")
    # trainer.extend(au_evaluator, trigger=val_interval, name='au_validation')
    # trainer.extend(Evaluator(validate_iter, model, converter=convert, device=-1), trigger=val_interval,
    #                name='accu_validation')
    # if args.with_crf:
    crf_evaluator = OpenCRFEvaluator(iterator=validate_iter, target=model, device=args.gpu)
    trainer.extend(crf_evaluator, trigger=val_interval, name="opencrf_val")

    trainer.run()

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