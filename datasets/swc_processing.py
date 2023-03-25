import numpy as np
from treelib import Tree
from collections import defaultdict
from swc_handler import parse_swc, write_swc, get_child_dict, get_index_dict


def is_in_box(x, y, z, imgshape):
    """
    imgshape must be in (z,y,x) order
    """
    if x < 0 or y < 0 or z < 0 or \
            x > imgshape[2] - 1 or \
            y > imgshape[1] - 1 or \
            z > imgshape[0] - 1:
        return False
    return True


def trim_out_of_box(tree_orig, imgshape, keep_candidate_points=True):
    """
    Trim the out-of-box leaves
    """
    # execute trimming
    child_dict = {}
    for leaf in tree_orig:
        if leaf[-1] in child_dict:
            child_dict[leaf[-1]].append(leaf[0])
        else:
            child_dict[leaf[-1]] = [leaf[0]]

    pos_dict = {}
    for i, leaf in enumerate(tree_orig):
        pos_dict[leaf[0]] = leaf

    tree = []
    for i, leaf in enumerate(tree_orig):
        idx, type_, x, y, z, r, p = leaf
        ib = is_in_box(x, y, z, imgshape)
        leaf = (idx, type_, x, y, z, r, p, ib)
        if ib:
            tree.append(leaf)
        elif keep_candidate_points:
            if p in pos_dict and is_in_box(*pos_dict[p][2:5], imgshape):
                tree.append(leaf)
            elif idx in child_dict:
                for ch_leaf in child_dict[idx]:
                    if is_in_box(*pos_dict[ch_leaf][2:5], imgshape):
                        tree.append(leaf)
                        break
    return tree


def swc_to_points(tree, imgshape, p_idx=-2):
        pos_dict = {}
        roots = []
        for i, leaf in enumerate(tree):
            pos_dict[leaf[0]] = leaf

        for i, leaf in enumerate(tree):
            if leaf[p_idx] not in pos_dict:
                roots.append(leaf[0])

        poses = []
        labels = []
        
        child_dict = get_child_dict(tree, p_idx_in_leaf=p_idx)
        for line in tree:
            idx, _, x, y, z, *_, par, ib = line
            x = x / imgshape[-1]
            y = y / imgshape[-2]
            z = z / imgshape[0]
            if idx in child_dict:
                cnum = len(child_dict[idx])
                if par == -1:
                    poses.append([z, y, x])
                    labels.append(1)
                elif cnum >= 2:
                    poses.append([z, y, x])
                    labels.append(2)
                elif idx in roots:
                    while ib == 0:
                        child_idx = child_dict[idx][0]
                        child_leaf = pos_dict[child_idx]
                        idx, _, x, y, z, *_, par, ib = child_leaf
                    x = x / imgshape[-1]
                    y = y / imgshape[-2]
                    z = z / imgshape[0]
                    poses.append([z, y, x])
                    labels.append(4)
            elif ib == 1:
                poses.append([z, y, x])
                labels.append(3)
            elif ib == 0:
                while ib == 0: 
                    par_leaf = pos_dict[par]
                    idx, _, x, y, z, *_, par, ib = par_leaf
                x = x / imgshape[-1]
                y = y / imgshape[-2]
                z = z / imgshape[0]
                poses.append([z, y, x])
                labels.append(4)

        return poses, labels


if __name__ == '__main__':
    swcfile = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/17302_18816.00_39212.03_2416.26.swc'
    tree = parse_swc(swcfile)
    sz = 50
    sy = 100
    sx = 100
    new_tree = []
    for leaf in tree:
        idx, type_, x, y, z, r, p = leaf
        x = x - sx
        y = y - sy
        z = z - sz
        new_tree.append((idx, type_, x, y, z, r, p))
    tree = trim_out_of_box(new_tree, imgshape=[32, 64, 64])
    print(len(tree))
    poses, labels = swc_to_points(tree, imgshape=[32,64,64])

    print(poses)
    print(labels)


