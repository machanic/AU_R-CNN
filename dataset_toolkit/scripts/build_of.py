__author__ = 'yjxiong'

import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool

import argparse

'''
./extract_gpu -f test.avi -x tmp/flow_x -y tmp/flow_y -i tmp/image -b 20 -t 1 -d 0 -s 1 -o dir

    test.avi: input video
    tmp: folder containing RGB images and optical flow images
    dir: output generated images to folder. if set to zip, will write images to zip files instead.
                        "{ f vidFile      | ex2.avi | filename of video }"
                        "{ x xFlowFile    | flow_x | filename of flow x component }"
                        "{ y yFlowFile    | flow_y | filename of flow x component }"
                        "{ i imgFile      | flow_i | filename of flow image}"
                        "{ b bound | 15 | specify the maximum of optical flow}"
                        "{ t type | 0 | specify the optical flow algorithm }"
                        "{ d device_id    | 0  | set gpu id}"
                        "{ s step  | 1 | specify the step for frame sampling}"
                        "{ o out | zip | output style}"
                        "{ w newWidth | 0 | output style}"
                        "{ h newHeight | 0 | output style}"
'''

def run_optical_flow(video_path, database, out_folder_path, dev_id=0, base_cmd="denseFlow_gpu"):
    seq_key = os.path.basename(video_path)[:os.path.basename(video_path).rindex(".")]
    if database == "BP4D":
        seq_key = seq_key.replace("_", "/")
    elif databases == "DISFA":
        seq_key = seq_key[:seq_key.rindex("_")]

    out_full_path = os.path.join(out_folder_path, seq_key)
    print(out_full_path + " " + "!!!")
    try:
        os.makedirs(out_full_path,exist_ok=True)
    except OSError:
        pass


    cmd = os.path.join(df_path + '/{}'.format(base_cmd))+' -f={} -o={} -b=60 -t=1 -d={} -s=1  -h=0 -w=0'.format(
        quote(video_path),  quote(out_full_path), dev_id)  # FIXME 删掉了-w 和 -h
    print(cmd)
    os.system(cmd)
    print('{} done'.format(seq_key))
    sys.stdout.flush()
    return True


def run_warp_optical_flow(video_path, out_folder_path, dev_id=0):
    return run_optical_flow(video_path, out_folder_path, dev_id, "extract_warp_gpu")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--num_worker", type=int, default=3)
    parser.add_argument("--flow_type", type=str, default='tvl1', choices=['tvl1', 'warp_tvl1'])
    parser.add_argument("--df_path", type=str, default='/home/machen/download2/denseFlow_gpu', help='path to the dense_flow toolbox')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'],
                        help='path to the dense_flow toolbox')
    parser.add_argument("--ext", type=str, default='mp4', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=8, help='number of GPU')
    parser.add_argument("--database", default="BP4D", help='BP4D/DISFA')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    flow_type = args.flow_type
    df_path = args.df_path
    out_format = args.out_format
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu

    os.makedirs(out_path, exist_ok=True)

    vid_list = glob.glob(src_path+'/*.'+ext)
    output_dirs = [out_path] * len(vid_list)
    databases = [args.database] * len(vid_list)
    print(len(vid_list))
    pool = Pool(num_worker)
    if flow_type == 'tvl1':
        pool.starmap(run_optical_flow, zip(vid_list, databases, output_dirs))
    # elif flow_type == 'warp_tvl1':
    #     pool.map(run_warp_optical_flow, zip(vid_list, output_dirs))
