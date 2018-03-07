if args.eval_mode:
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        test_data = AUDataset(database=args.database, fold=args.fold,
                              split_name='test', split_index=args.split_idx, mc_manager=mc_manager,
                              use_lstm=args.use_lstm, train_all_data=False, prefix=args.prefix,
                              pretrained_target=args.pretrained_target)
        test_data = TransformDataset(test_data,
                                     Transform(faster_rcnn, mirror=False, shift=False, use_lstm=args.use_lstm))
        if args.proc_num == 1:
            test_iter = SerialIterator(test_data, 1, repeat=False, shuffle=True)
        else:
            test_iter = MultiprocessIterator(test_data, batch_size=1, n_processes=args.proc_num,
                                             repeat=False, shuffle=True,
                                             n_prefetch=10, shared_mem=10000000)

        gpu = int(args.gpu) if "," not in args.gpu else int(args.gpu[:args.gpu.index(",")])
        chainer.cuda.get_device_from_id(gpu).use()
        faster_rcnn.to_gpu(gpu)
        evaluator = AUEvaluator(test_iter, faster_rcnn,
                                lambda batch, device: concat_examples(batch, device, padding=-99),
                                args.database, device=gpu)
        observation = evaluator.evaluate()
        with open(args.out + os.sep + "evaluation_result.json", "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()
    return

