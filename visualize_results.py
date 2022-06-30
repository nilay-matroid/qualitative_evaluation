# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com

Adapted by
@author: Nilay Pande
@contact: nilay017@gmail.com
"""

import argparse
import shutil
import sys
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from visualizer import Visualizer
from datasets import load_accumulated_info_of_dataset

sys.path.append('.')

def get_parser():
    parser = argparse.ArgumentParser(description="Visualization of person reid results given features")
    parser.add_argument(
        "--root_folder",
        default="../Datasets",
        type=str,
        help="Root folder containing datasets"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--input_dir",
        default="./cache",
        type=str,
        help="Input root directory containing saved feats, pids, camids and possibly imgpaths",
    )
    parser.add_argument(
        "--sep_query_gallery",
        action='store_true',
        help="if query and gallery features are separeted into two distinct folders - \"query\" and \"gallery\" "
    )
    parser.add_argument(
        "--use_eval_set",
        action='store_true',
        help="if query and gallery features are separeted into two distinct folders - \"query\" and \"gallery\" "
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Output directory to save visualization result.",
    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Sets verbosity"
    )
    parser.add_argument(
        "--keep_pid_camid",
        action='store_true',
        help="Doesnot remove gallery images with same pid and camid if flag is used"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        type=int,
        help=
        "number of query images to be visualized",
    )
    parser.add_argument(
        "--label-sort",
        default="descending",
        type=str,
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        type=int,
        help="maximum number of rank list to be visualized",
    )
    return parser


def _get_imagedata_info(data):
    feats, pids, camids = data
    num_pids = np.unique(pids).shape[0]
    num_cams = np.unique(camids).shape[0]
    num_imgs = feats.shape[0]
    return num_pids, num_imgs, num_cams

def _print_dataset_statistics_movie(query, gallery):
    num_query_pids, num_query_imgs, _ = _get_imagedata_info(query)
    num_gallery_pids, num_gallery_imgs, _ = _get_imagedata_info(gallery)

    test_or_eval = "test"

    print("Dataset statistics:")
    print("  --------------------------------------")
    print("  subset         | # ids     | # images")
    print("  --------------------------------------")
    print("  query   ({})       | {:5d}     | {:8d}".format(test_or_eval, num_query_pids, num_query_imgs))
    print("  gallery ({})      | {:5d}     | {:8d}".format(test_or_eval, num_gallery_pids, num_gallery_imgs))

def load_saved_feat(dir=None):
    assert os.path.isdir(dir), f"{dir} doesn't exist"
    feat_file = os.path.join(dir, "feat.npy")
    pid_file = os.path.join(dir, "pid.npy")
    camid_file = os.path.join(dir, "camid.npy")
    imgpath_file = os.path.join(dir, "imgpath.npy")

    assert os.path.isfile(feat_file), "Features not found"
    assert os.path.isfile(pid_file), "Pids not found"
    assert os.path.isfile(camid_file), "Camids not found"

    if os.path.isfile(imgpath_file):
        print("File containing saved image paths found")
        return (np.load(feat_file, allow_pickle=True).astype(np.float32),\
             np.load(pid_file, allow_pickle=True).astype(np.float32), np.load(camid_file, allow_pickle=True), np.load(imgpath_file, allow_pickle=True).tolist())    

    return (np.load(feat_file, allow_pickle=True).astype(np.float32),\
         np.load(pid_file, allow_pickle=True).astype(np.float32), np.load(camid_file, allow_pickle=True), None)


if __name__ == '__main__':
   
    parser = get_parser()
    args = parser.parse_args()

    assert os.path.isdir(args.input_dir), "Oops no input cache directory found"

    _, query, gallery = load_accumulated_info_of_dataset(args.root_folder, args.dataset_name, use_eval_set=args.use_eval_set, verbose=args.verbose)
    num_query = pd.DataFrame(query)["image_file_path"].nunique()
    test_dataset =  query + gallery

    # Load saved features
    if not args.sep_query_gallery:
        feats, pids, camids, imgpaths = load_saved_feat(os.path.join(args.input_dir, args.dataset_name))
        q_feats = feats[:num_query]
        g_feats = feats[num_query:]
        q_pids = np.asarray(pids[:num_query])
        g_pids = np.asarray(pids[num_query:])
        q_camids = np.asarray(camids[:num_query])
        g_camids = np.asarray(camids[num_query:])
    else:
        q_feats, q_pids, q_camids, q_imgpaths = load_saved_feat(os.path.join(args.input_dir, args.dataset_name, "query"))
        g_feats, g_pids, g_camids, g_imgpaths = load_saved_feat(os.path.join(args.input_dir, args.dataset_name, "gallery"))

        if q_imgpaths is not None and g_imgpaths is not None:
            imgpaths = q_imgpaths + g_imgpaths


    if args.verbose:
        print("Printing loaded features stats ....")
        _print_dataset_statistics_movie((q_feats, q_pids, q_camids), (g_feats, g_pids, g_camids))

    # import pdb
    # pdb.set_trace()

    visualizer = Visualizer(test_dataset, imgpaths)

    # remove_same_pid_camid will be True by default
    remove_same_pid_camid = not args.keep_pid_camid
    visualizer.get_model_output(q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank=args.max_rank, remove_same_pid_camid=remove_same_pid_camid)

    # print("Start saving ROC curve ...")
    # fpr, tpr, pos, neg = visualizer.vis_roc_curve(args.output)
    # visualizer.save_roc_info(args.output, fpr, tpr, pos, neg)
    # print("Finish saving ROC curve!")

    output_dir = os.path.join(args.output_dir, args.dataset_name)

    print("Removing old output files ... ")
    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)

    print("Saving rank list result ...")
    query_indices = visualizer.vis_rank_list(output_dir, args.vis_label, args.num_vis, args.label_sort, args.max_rank)
    print("Finish saving rank list results!")
