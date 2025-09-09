from setup import *
import copy
from configure import *
import warnings
warnings.filterwarnings('ignore')

class Node:
    """
        1. depth: 节点深度
        2. idx: 当前深度的第几个节点
        3. parent_idx: 父节点是上一层的第几个节点
        4. name: 节点详情
        5. child_num: 节点拥有几个孩子节点(单运算还是双运算)
        6. child_st: 下一层从第几个节点开始是当前的孩子节点
        7. var: 节点variable/operation
        8. cache: 保留运算至今的数据
        9. status: 初始化为child_num，用于记录遍历状态
        10. full: 完整信息，以OP或VAR形式表示
    """
    def __init__(self, depth, idx, parent_idx, name, full, child_num, child_st, var):
        self.depth = depth
        self.idx = idx
        self.parent_idx = parent_idx

        self.name = name
        self.child_num = child_num
        self.child_st = child_st
        self.status = self.child_num
        self.var = var
        self.full = full
        self.cache = copy.deepcopy(var)

    def __str__(self): # 提供节点详情
        return self.name

    def reset_status(self): # 初始化status
        self.status = self.child_num

class Tree: #对应于pde中的一个term
    def __init__(self, max_depth, p_var):
        self.max_depth = max_depth
        self.tree = [[] for i in range(max_depth)]
        self.preorder, self.inorder = None, None

        root = ROOT[np.random.randint(0, len(ROOT))] # 随机产生初始root（一种OPS）# e.g. ['sin', 1, np.sin], ['*', 2, np.multiply] 
        node = Node(depth=0, idx=0, parent_idx=None, name=root[0], var=root[2], full=root,
                    child_num=int(root[1]), child_st=0) # 设置初始节点Node
        self.tree[0].append(node) # 初始节点

        depth = 1
        while depth < max_depth:
            next_cnt = 0 #child_st=next_cnt， child_st: 下一层从第几个节点开始是当前的孩子节点
            # 对应每一个父节点都要继续生成他们的子节点
            for parent_idx in range(len(self.tree[depth - 1])): #一个tree中某个depth处的node的子节点可以是多个，因此有可能在某个深度处存在多个node
                parent = self.tree[depth - 1][parent_idx] # 提取出对应深度处的对应操作符（某个node）
                if parent.child_num == 0: # 如果当前node没有子节点，则跳过当前循环的剩余语句，然后继续进行下一轮循环
                    continue
                for j in range(parent.child_num):
                    # rule 1: parent var为d 且j为1时，必须确保右子节点为x
                    if parent.name in {'d', 'd^2'} and j == 1: # j == 0 为d的左侧节点，j == 1为d的右侧节点
                        node = den[np.random.randint(0, len(den))] # 随机产生一个微分运算的denominator，一般是xyt
                        node = Node(depth=depth, idx=len(self.tree[depth]), parent_idx=parent_idx, name=node[0],
                                    var=node[2], full=node, child_num=int(node[1]), child_st=None)
                        self.tree[depth].append(node)
                    # rule 2: 最底一层必须是var，不能是op
                    elif depth == max_depth - 1:
                        node = VARS[np.random.randint(0, len(VARS))]
                        node = Node(depth=depth, idx=len(self.tree[depth]), parent_idx=parent_idx, name=node[0],
                                    var=node[2], full=node, child_num=int(node[1]), child_st=None)
                        self.tree[depth].append(node)
                    else:
                    # rule 3: 不是最底一层，p_var概率产生var，如果产生var，则child_st为None。当产生的是ops的时候，产生对应node时对应的child_st会通过计算获得。以便对应于下一层中该ops对应的子节点。
                        if np.random.random() <= p_var:
                            node = VARS[np.random.randint(0, len(VARS))]
                            node = Node(depth=depth, idx=len(self.tree[depth]), parent_idx=parent_idx, name=node[0],
                                        var=node[2], full=node, child_num=int(node[1]), child_st=None)
                            self.tree[depth].append(node)
                        else:
                            node = OPS[np.random.randint(0, len(OPS))]
                            node = Node(depth=depth, idx=len(self.tree[depth]), parent_idx=parent_idx, name=node[0],
                                        var=node[2], full=node, child_num=int(node[1]), child_st=next_cnt)
                            next_cnt += node.child_num
                            self.tree[depth].append(node)
            depth += 1

        ret = []
        dfs(ret, self.tree, depth=0, idx=0)
        self.preorder = ' '.join([x for x in ret])
        model_tree = copy.deepcopy(self.tree)
        self.inorder = tree2str_merge(model_tree)


    def mutate(self, p_mute): 
        global see_tree
        see_tree = copy.deepcopy(self.tree)
        depth = 1
        while depth < self.max_depth:
            next_cnt = 0
            idx_this_depth = 0  # 这个深度第几个节点
            for parent_idx in range(len(self.tree[depth - 1])):
                parent = self.tree[depth - 1][parent_idx]
                if parent.child_num == 0:
                    continue
                for j in range(parent.child_num):  # parent 的第j个子节点
                    not_mute = np.random.choice([True, False], p=([1 - p_mute, p_mute]))
                    # rule 1: 不突变则跳过
                    if not_mute:
                        next_cnt += self.tree[depth][parent.child_st + j].child_num
                        continue
                    # 当前节点的类型
                    current = self.tree[depth][parent.child_st + j]
                    temp = self.tree[depth][parent.child_st + j].name
                    num_child = self.tree[depth][parent.child_st + j].child_num  # 当前变异节点的子节点数
                    # print('mutate!')
                    if num_child == 0: # 叶子节点
                        node = VARS[np.random.randint(0, len(VARS))] # rule 2: 叶节点必须是var，不能是op
                        # Adjust indexing for list of lists
                        while node[0] == temp or (parent.name in {'d', 'd^2'} and node[0] not in [item[0] for item in den]): # rule 3: 如果编译前后结果重复，或者d的节点不在den中（即出现不能求导的对象），则重新抽取
                            if parent.name in {'d', 'd^2'} and node[0] == 'x': # simple_mode中，遇到对于x的导数，直接停止变异
                                break                            
                            node = VARS[np.random.randint(0, len(VARS))] # 重新抽取一个vars
                        new_node = Node(depth=depth, idx=idx_this_depth, parent_idx=parent_idx, name=node[0],
                                    var=node[2], full=node, child_num=int(node[1]), child_st=None)
                        self.tree[depth][parent.child_st + j] = new_node #替换成变异的节点
                    else: # 非叶子节点
                        if num_child == 1:
                            node = OP1[np.random.randint(0, len(OP1))]
                            while node[0] == temp:  # 避免重复
                                node = OP1[np.random.randint(0, len(OP1))]
                        elif num_child == 2:
                            node = OP2[np.random.randint(0, len(OP2))]
                            right = self.tree[depth + 1][current.child_st + 1].name
                            # Adjust indexing for list of lists
                            while node[0] == temp or (node[0] in {'d', 'd^2'} and right not in [item[0] for item in den]): # rule 4: 避免重复，避免生成d以打乱树结构（新d的右子节点不是x）
                                node = OP2[np.random.randint(0, len(OP2))]
                        else:
                            raise NotImplementedError("Error occurs!")

                        new_node = Node(depth=depth, idx=idx_this_depth, parent_idx=parent_idx, name=node[0],
                                    var=node[2], full=node, child_num=int(node[1]), child_st=next_cnt)
                        next_cnt += new_node.child_num
                        self.tree[depth][parent.child_st + j] = new_node
                    idx_this_depth += 1
            depth += 1

        ret = []
        dfs(ret, self.tree, depth=0, idx=0)
        self.preorder = ' '.join([x for x in ret])
        model_tree = copy.deepcopy(self.tree)
        self.inorder = tree2str_merge(model_tree)
    

def dfs(ret, a_tree, depth, idx): #辅助前序遍历，产生一个描述这个tree的名称序列（ret）
    # print(depth, idx)  # 深度优先遍历的顺序
    node = a_tree[depth][idx]
    ret.append(node.name) # 记录当前操作
    for ix in range(node.child_num):
        if node.child_st is None:
            continue
        dfs(ret, a_tree, depth+1, node.child_st + ix) #进入下一层中下一个节点对应的子节点

def tree2str_merge(a_tree):
    for i in range(len(a_tree) - 1, 0, -1):
        for node in a_tree[i]:
            if node.status == 0:
                if a_tree[node.depth-1][node.parent_idx].status == 1:
                    if a_tree[node.depth-1][node.parent_idx].child_num == 2:
                        a_tree[node.depth-1][node.parent_idx].name = a_tree[node.depth-1][node.parent_idx].name + ' ' + node.name + ')'
                    else:
                        a_tree[node.depth-1][node.parent_idx].name = '( ' + a_tree[node.depth-1][node.parent_idx].name + ' ' + node.name + ')'
                elif a_tree[node.depth-1][node.parent_idx].status > 1:
                    a_tree[node.depth-1][node.parent_idx].name = '(' + node.name + ' ' + a_tree[node.depth-1][node.parent_idx].name
                a_tree[node.depth-1][node.parent_idx].status -= 1
    return a_tree[0][0].name


class PDE:
    def __init__(self, depth, max_width, p_var):
        self.depth = depth
        self.p_var = p_var
        #self.W = np.random.randint(max_width)+1  # 1 -- width
        self.W = (np.random.randint(max_width // 2) + 1) * 2 #even number >= 2
        self.elements = []
        for i in range(0, self.W):
            # 产生W个tree，也就是W个项
            one_tree = Tree(depth, p_var)
            self.elements.append(one_tree)

    def mutate(self, p_mute):
        for i in range(0, self.W):  # 0 -- W-1
            self.elements[i].mutate(p_mute)

    def replace(self): 
        one_tree = Tree(self.depth, self.p_var)# 直接产生一个新的tree，替换pde中的一项
        ix = np.random.randint(self.W)  # 0 -- W-1
        if len(self.elements) == 0:
            NotImplementedError('replace error')
        self.elements[ix] = one_tree

    def replace_selected(self, terms): 
        trees = terms[0]
        prob = terms[1]
        if len(trees)==0:
            one_tree = Tree(self.depth, self.p_var)
        else:
            one_tree = np.random.choice(trees, p=prob)
        ix = np.random.randint(self.W)  # 0 -- W-1
        if len(self.elements) == 0:
            NotImplementedError('replace error')
        self.elements[ix] = one_tree

    def visualize(self): # 写出SGA产生的项的形式，包含产生的所有项，未去除系数小的项。
        name = ''
        for i in range(len(self.elements)):
            if i != 0:
                name += '+'
            name += self.elements[i].inorder
        return name
     
    def concise_visualize(self):
        # 1) Copy the term‐trees and get raw coefficients
        elements = copy.deepcopy(self.elements)
        elements, raw_coeffs = evaluate_mse(elements, True)
        coeffs = np.asarray(raw_coeffs).flatten()

        # 2) Zero out any infs or overly large values
        finite_mask = np.isfinite(coeffs)
        coeffs[~finite_mask] = 0.0
        large_mask = np.abs(coeffs) > 1e4
        coeffs[large_mask] = 0.0

        # 3) Now split into drift vs diffusion
        half       = len(coeffs) // 2
        drift_c    = coeffs[:half]
        diffusion_c= coeffs[half:]
        drift_t    = elements[:half]
        diffusion_t= elements[half:]

        # 4) Build the string parts, ignoring tiny (<1e-4) or now-zeroed coefficients
        drift_parts = [
            f"{round(float(c),4)}{t.inorder}"
            for c, t in zip(drift_c, drift_t)
            if abs(c) >= 1e-4
        ]
        diffusion_parts = [
            f"{round(float(c),4)}{t.inorder}"
            for c, t in zip(diffusion_c, diffusion_t)
            if abs(c) >= 1e-4
        ]

        # 5) Combine into the final name
        drift_str     = " + ".join(drift_parts)     if drift_parts     else ""
        diffusion_str = " + ".join(diffusion_parts) if diffusion_parts else ""

        if drift_str and diffusion_str:
            name = f"Drift: {drift_str} ; Diffusion: {diffusion_str}"
        elif drift_str:
            name = f"Drift: {drift_str}"
        else:
            name = f"Diffusion: {diffusion_str}"

        # 6) (Optional) show cleaned coeff arrays
        print("Drift coeffs (cleaned):    ", drift_c)
        print("Diffusion coeffs (cleaned):", diffusion_c)

        return name


    

def bin_features(feature_matrix, x_grid):
    """Bin feature matrix according to KM bin structure"""
    bin_edges = config.KM_bin_edges
    valid_indices = config.KM_central_valid_indices
    
    bin_indices = np.digitize(x_grid, bin_edges) - 1
    n_bins = len(valid_indices)
    n_features = feature_matrix.shape[1]
    binned = np.zeros((n_bins, n_features))
    
    for i, bin_idx in enumerate(valid_indices):
        in_bin = (bin_indices == bin_idx)
        if np.any(in_bin):
            binned[i] = np.mean(feature_matrix[in_bin], axis=0)
    return binned

def evaluate_mse(a_pde, is_term=False, metric=None):
    """
    Evaluate a PDE (or list of term‐trees) against KM drift/diffusion targets
    by evaluating directly on the trimmed KM bin centers (config.x).

    Returns:
      - if is_term=False: (AIC_total, weights_concat)
      - if is_term=True: (terms_list, weights_concat)
    """
    import numpy as np
    import configure as config
    from setup import Evaluate2
    if metric is None:
        metric = getattr(config, 'error_metric', 'mse')
        
    # 1) determine the list of term‐trees to evaluate
    terms = a_pde.elements if not is_term else a_pde

    # 2) number of evaluation points = number of trimmed bins
    n_pts = config.x.size

    # 3) allocate storage: rows=bins, cols=terms
    terms_values = np.zeros((n_pts, len(terms)))

    # 4) evaluate each tree at each bin center
    for ix, term in enumerate(terms):
        tree = term.tree
        # roll cached values from leaves up to root
        for depth in range(len(tree) - 1, 0, -1):
            for node in tree[depth]:
                if node.parent_idx is None:
                    continue
                parent = tree[depth - 1][node.parent_idx]
                if parent.child_num == 1:
                    # unary operator
                    parent.cache = parent.var(node.cache)
                else:
                    # binary operator (possibly derivative)
                    if parent.name in {'d', 'd^2'}:
                        axis_name = tree[depth][parent.child_st + 1].name
                        denom = config.dx if axis_name == 'x' else config.dt
                        parent.cache = parent.var(
                            tree[depth][parent.child_st].cache,
                            denom,
                            axis_name
                        )
                    else:
                        parent.cache = parent.var(
                            tree[depth][parent.child_st].cache,
                            tree[depth][parent.child_st + 1].cache
                        )
        # store nonzero outputs
        root_cache = tree[0][0].cache.ravel()
        if not np.all(root_cache == 0):
            terms_values[:, ix] = root_cache

    # 5) split into drift and diffusion feature matrices
    half = terms_values.shape[1] // 2
    drift_val = terms_values[:, :half]
    diff_val  = terms_values[:, half:]

    # 6) pull out KM targets (already length n_pts)
    Ut_d = config.drift_target.flatten()
    Ut_f = config.diffusion_target.flatten()

    # 7) drop any bins with NaN or Inf in either feature set
    mask = np.isfinite(drift_val).all(axis=1) & np.isfinite(diff_val).all(axis=1)
    drift_val = drift_val[mask, :]
    diff_val  = diff_val[mask, :]
    Ut_d      = Ut_d[mask]
    Ut_f      = Ut_f[mask]

    w_all, err_d, err_f, aic_total, aic_d, aic_f = Evaluate2(
        drift_val,
        diff_val,
        Ut_d,
        Ut_f,
        lam=1e-6,#0,
        d_tol=0.5,#1,
        AIC_ratio=config.aic_ratio,
        metric=metric
    )

    if not is_term:
        return aic_total, w_all
    else:
        return terms, w_all

