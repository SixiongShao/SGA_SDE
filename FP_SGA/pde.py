from setup import *
import copy
from configure import *
import warnings
#from KM_SGA.setup_KM import *
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
    
    @classmethod  
    def from_custom_structure(cls, nodes):
        if not nodes:
            raise ValueError("The node list is empty.")

        # Determine the maximum depth of the tree
        max_depth = max(node['depth'] for node in nodes) + 1

        # Initialize the tree with empty lists for each depth
        tree = [[] for _ in range(max_depth)]

        # Create the nodes and populate the tree
        for node_info in nodes:
            node = Node(
                depth=node_info['depth'],
                idx=node_info['idx'],
                parent_idx=node_info['parent_idx'],
                name=node_info['name'],
                full=node_info['full'],
                child_num=node_info['child_num'],
                child_st=node_info['child_st'],
                var=node_info['var']
            )
            tree[node_info['depth']].append(node)

        # Ensure child_st is set correctly for each node with children
        for depth in range(max_depth - 1):
            next_cnt = 0
            for parent_idx, parent in enumerate(tree[depth]):
                if parent.child_num > 0:
                    parent.child_st = next_cnt
                    next_cnt += parent.child_num

        # Create an instance of the Tree class
        tree_instance = cls(max_depth=max_depth, p_var=0)
        tree_instance.tree = tree

        # Generate preorder and inorder strings
        ret = []
        dfs(ret, tree_instance.tree, depth=0, idx=0)
        tree_instance.preorder = ' '.join([x for x in ret])
        model_tree = copy.deepcopy(tree_instance.tree)
        tree_instance.inorder = tree2str_merge(model_tree)

        return tree_instance

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
 
    # def concise_visualize(self):  # For no default term # 写出所有项的形式，包含固定候选集和SGA，且包含系数。会区分是来自于固定候选集的还是来自于SGA生成的候选集的。如果是来自于SGA生成的候选集，需要用inorder来写出可理解的项。
    #     name = ''
    #     elements = copy.deepcopy(self.elements)
    #     elements, coefficients = evaluate_mse(elements, True)
    #     coefficients = coefficients[:, 0]
    #     for i in range(len(coefficients)):
    #         if np.abs(coefficients[i]) < 1e-4:  # Ignore coefficients that are too small
    #             continue
    #         if i != 0 and name != '':
    #             name += ' + '
    #         name += str(round(np.real(coefficients[i]), 4))
    #         name += elements[i].inorder  # element is the candidate set generated by SGA
    #     print("Coefficients: ", coefficients)
    #     return name     
    def concise_visualize(self):  # For no default term FP form
        name = ''
        drift_name = ''
        diffusion_name = ''
        elements = copy.deepcopy(self.elements)
        elements, coefficients = evaluate_mse(elements, True)
        coefficients = coefficients[:, 0]
        
        # Split ALL terms into drift (first half) and diffusion (second half)
        n_total_terms = len(coefficients)
        drift_terms = elements[:n_total_terms//2]
        drift_coeffs = coefficients[:n_total_terms//2]
        diffusion_terms = elements[n_total_terms//2:]
        diffusion_coeffs = coefficients[n_total_terms//2:]
        
        # Filter out small coefficients (< 1e-4) for drift terms
        drift_parts = []
        for coeff, term in zip(drift_coeffs, drift_terms):
            if np.abs(coeff) >= 1e-4:
                drift_parts.append(f"{round(np.real(coeff), 4)}{term.inorder}")
        
        # Filter out small coefficients (< 1e-4) for diffusion terms
        diffusion_parts = []
        for coeff, term in zip(diffusion_coeffs, diffusion_terms):
            if np.abs(coeff) >= 1e-4:
                diffusion_parts.append(f"{round(np.real(coeff), 4)}{term.inorder}")
        
        # Combine drift parts (if any)
        if drift_parts:
            drift_name = " + ".join(drift_parts)
        
        # Combine diffusion parts (if any)
        if diffusion_parts:
            diffusion_name = " + ".join(diffusion_parts)
        
        # Final formatting
        if drift_name and diffusion_name:
            name = f"Drift: {drift_name} ; Diffusion: {diffusion_name}"
        elif drift_name:
            name = f"Drift: {drift_name}"
        elif diffusion_name:
            name = f"Diffusion: {diffusion_name}"
        
        #print(f"Total terms: {n_total_terms}, Drift terms: {len(drift_terms)}, Diffusion terms: {len(diffusion_terms)}")
        print(f"Drift coefficients: {drift_coeffs}")
        print(f"Diffusion coefficients: {diffusion_coeffs}")
        return name
    
    @classmethod
    def from_custom_structure(cls, custom_trees, p_var=0.5):
        max_depth = max(tree.max_depth for tree in custom_trees)
        max_width = len(custom_trees)
        pde = cls(max_depth, max_width, p_var)
        pde.elements = custom_trees
        return pde


def evaluate_mse(a_pde, is_term=False, FP=True,KM=False):#FP:fokker planck, KM:kramers moyal
    if is_term:
        terms = a_pde
    else:
        terms = a_pde.elements
    terms_values = np.zeros((u.shape[0] * u.shape[1], len(terms)))
    delete_ix = []
    for ix, term in enumerate(terms):
        tree_list = term.tree
        max_depth = len(tree_list)

        # 先搜索倒数第二层，逐层向上对数据进行运算直到顶部；排除底部空层
        for i in range(2, max_depth+1):
            # 如果下面一层是空的，说明这一层肯定不是非空的倒数第二层
            if len(tree_list[-i+1]) == 0:
                continue
            else: # 这一层是非空至少倒数第二层，一个一个结点看过去
                for j in range(len(tree_list[-i])):
                    # 如果这一结点没有孩子，继续看右边的结点有没有
                    if tree_list[-i][j].child_num == 0:
                        continue

                    # 这一结点有一个孩子，用自己的运算符对孩子的cache进行操作
                    elif tree_list[-i][j].child_num == 1:
                        child_node = tree_list[-i+1][tree_list[-i][j].child_st]
                        tree_list[-i][j].cache = tree_list[-i][j].cache(child_node.cache)
                        child_node.cache = child_node.var  # 重置

                    # 这一结点有一两个孩子，用自己的运算符对两孩子的cache进行操作
                    elif tree_list[-i][j].child_num == 2:
                        child1 = tree_list[-i+1][tree_list[-i][j].child_st]
                        child2 = tree_list[-i+1][tree_list[-i][j].child_st+1]

                        if tree_list[-i][j].name in {'d', 'd^2'}:
                            what_is_denominator = child2.name
                            if what_is_denominator == 't':
                                tmp = dt
                            elif what_is_denominator == 'x':
                                tmp = dx
                            else:
                                raise NotImplementedError()

                            if not isfunction(tree_list[-i][j].cache):
                                pdb.set_trace()
                                tree_list[-i][j].cache = tree_list[-i][j].var

                            tree_list[-i][j].cache = tree_list[-i][j].cache(child1.cache, tmp, what_is_denominator)

                        else:
                            if isfunction(child1.cache) or isfunction(child2.cache):
                                pdb.set_trace()
                            tree_list[-i][j].cache = tree_list[-i][j].cache(child1.cache, child2.cache)
                        child1.cache, child2.cache = child1.var, child2.var  # 重置

                    else:
                        NotImplementedError()

        if not any(tree_list[0][0].cache.reshape(-1)):  # 如果全是0，无法收敛且无意义
            delete_ix.append(ix)
            tree_list[0][0].cache = tree_list[0][0].var  # 重置缓冲池
            # print('0')
            # pdb.set_trace()
        else:
            # if ix == len(terms)-1: # If this is the last term, multiply its values by a standard normal distribution N(0,1)
            #     random_noise = np.random.normal(0,1,size=tree_list[0][0].cache.shape)* np.sqrt(dt)
            #     #random_noise = du #testing
            #     tree_list[0][0].cache *= random_noise
                
            terms_values[:, ix:ix+1] = tree_list[0][0].cache.reshape(-1, 1)  # 把归并起来的该term记录下来
            tree_list[0][0].cache = tree_list[0][0].var  # 重置缓冲池
            # print('not 0')
            # pdb.set_trace()

    move = 0
    for ixx in delete_ix:
        if is_term:
            terms.pop(ixx - move)
        else:
            a_pde.elements.pop(ixx-move)
            a_pde.W -= 1  # 实际宽度减一
        terms_values = np.delete(terms_values, ixx-move, axis=1)
        move += 1  # pop以后index左移

    ### Fokker Planck Edition ###
    n, m = u.shape  # Grid dimensions
    if FP==True:
        n_terms = terms_values.shape[1]  # Number of terms after removal
        
        
        if n_terms > 0:  # Only transform if terms exist
            half_idx = n_terms // 2  # Split point for halves
            
            # Process first half: multiply by u and first derivative w.r.t x
            for i in range(half_idx):
                term_vals = terms_values[:, i].reshape(n, m)
                term_vals = term_vals * u *(-1)  # Multiply by -u
                term_vals = Diff(term_vals, dx, 'x')  # First derivative
                terms_values[:, i] = term_vals.ravel()
            
            # Process second half: multiply by 1/2*u and second derivative w.r.t x
            for i in range(half_idx, n_terms):
                term_vals = terms_values[:, i].reshape(n, m)
                term_vals = 0.5 * u * term_vals  # Multiply by 1/2*u
                term_vals = Diff2(term_vals, dx, 'x')  # Second derivative
                terms_values[:, i] = term_vals.ravel()
        
        #drift_val = terms_values[:, :half_idx]     # First half
        #diffusion_val = terms_values[:, half_idx:]  # Second half

    # 检查是否存在inf或者nan，或者terms_values是否被削没了
    if False in np.isfinite(terms_values) or terms_values.shape[1] == 0:
        # pdb.set_trace()
        error = np.inf
        aic = np.inf
        w = 0

    else:
        if KM == True:
            w, loss, mse, aic = Train(terms_values, ut.reshape(n * m, 1), 0, 1, aic_ratio)
            #w, loss, mse, aic, aic_drift, aic_diffusion = Evaluate2(drift_val,diffusion_val, drift_target.reshape(-1, 1),diffusion_target.reshape(-1, 1), 0, 1, aic_ratio)
        else:
            w, loss, mse, aic = Train(terms_values, ut.reshape(n * m, 1), 0, 1, aic_ratio)

    if is_term:
        return terms, w
    else:
        return aic, w 


if __name__ == '__main__':
    pde = PDE(depth=6, max_width=5, p_var=0.5)
    print(pde.visualize())
    print(pde.concise_visualize())

