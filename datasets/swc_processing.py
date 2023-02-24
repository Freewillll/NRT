import numpy as np
from treelib import Tree
from collections import defaultdict
from swc_handler import parse_swc, write_swc, get_child_dict


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


def swc_to_forest(tree, imgshape, p_idx=-2):
        pos_dict = {}
        seq_list = []
        level_dict = defaultdict(list)
        roots = []
        for i, leaf in enumerate(tree):
            pos_dict[leaf[0]] = leaf

        for i, leaf in enumerate(tree):
            if leaf[p_idx] not in pos_dict:
                roots.append(leaf[0])

        def dfs(idx, level, child_dict, tree):
            # 1 root, 2 branching point, 3 tip node, 4 boundary point, 5 other node
            leaf = pos_dict[idx]
            x, y, z, r = leaf[2:6]
            tag = 5
            if idx not in child_dict:
                *_, ib = leaf
                if idx in roots:
                    tag = 1
                elif ib == 1:
                    tag = 3
                    level += 1
                elif ib == 0:
                    tag = 4
                    level += 1

                if tag == 1 or tag == 3 or tag == 4:
                    level_dict[level].append(idx)
                if idx in roots:
                    tree.create_node(tag=tag, identifier=idx, data=(z, y, x))
                else:
                    tree.create_node(tag=tag, identifier=idx, parent=leaf[p_idx], data=(z, y, x))
                return
            else:
                cnum = len(child_dict[idx])
                if idx in roots:
                    tag = 1
                elif cnum == 1:
                    tag = 5
                elif cnum >= 2:
                    tag = 2
                    level += 1

                if tag == 1 or tag == 2:
                    level_dict[level].append(idx)
                if idx in roots:
                    tree.create_node(tag=tag, identifier=idx, data=(z, y, x))
                else:
                    tree.create_node(tag=tag, identifier=idx, parent=leaf[p_idx], data=(z, y, x))
                for cidx in child_dict[idx]:
                    dfs(cidx, level, child_dict, tree)

        Trees = []
        child_dict = get_child_dict(tree, p_idx_in_leaf=p_idx)
        for idx in roots:
            tree = Tree()
            seq = []
            dfs(idx, 0, child_dict, tree)
            sorted(level_dict)
            for key in level_dict:
                seq_item = []
                for idx in level_dict[key]:
                    node = tree.get_node(idx)
                    pos = np.clip(list(node.data), [0,0,0], [i-1 for i in imgshape])
                    seq_item.append(list(pos) + [node.tag])
                seq.append(seq_item)
            seq_list.append(seq)
            Trees.append(tree)
            level_dict.clear()
        return seq_list


if __name__ == '__main__':
    swcfile = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/18457_26188.42_11641.02_5421.82.swc'
    tree = parse_swc(swcfile)
    sz = 10
    sy = 10
    sx = 10
    new_tree = []
    for leaf in tree:
        idx, type_, x, y, z, r, p = leaf
        x = x - sx
        y = y - sy
        z = z - sz
        new_tree.append((idx, type_, x, y, z, r, p))
    
    tree = trim_out_of_box(tree, imgshape=[32, 64, 64])
    print(len(tree))
    seq_list = swc_to_forest(tree)
    print(len(seq_list[0]))
    print(seq_list[0])


