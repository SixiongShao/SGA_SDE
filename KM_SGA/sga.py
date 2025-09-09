from pde import *
import warnings
import sys
warnings.filterwarnings('ignore')


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class SGA:  
    def __init__(self, num, depth, width, p_var, p_mute, p_rep, p_cro):
        # num: pool里PDE的数量
        # depth: 每个PDE的term的最大深度
        # width: 每个PDE所含term的最大数量
        # p_var: 生成树时节点为u/t/x而不是运算符的概率
        # p_rep: 将（所有）pde某一项重新生成以替换原项的概率
        # p_mute: PDE的树结构里每个节点的突变概率
        # p_cro: 不同PDE之间交换term的概率
        self.num = num
        self.p_mute = p_mute
        self.p_cro = p_cro
        self.p_rep = p_rep
        self.eqs = []#PDEs in pool 1
        self.mses = []
        self.ratio = 1
        self.repeat_cross = 0
        self.repeat_change = 0
        self.mut_num = 0
        self.rep_num = 0
        self.cross_num = 0
        
        self.eqs2 = [] #Pool 2
        self.mses2 = []
        self.eqs3 = [] # Pool 3
        self.mses3 = []
        
        print('Creating the original pdes in the pool 1...')
        for i in range(num*self.ratio): # 循环产生num个pde
            a_pde = PDE(depth, width, p_var)
            a_err, a_w = evaluate_mse(a_pde)
            pde_lib.append(a_pde)
            err_lib.append((a_err, a_w))
            while a_err < -100 or a_err == np.inf:  # MSE太小则直接去除，to avoid u d t
                print(a_err)
                a_pde = PDE(depth, width, p_var)
                a_err, a_w = evaluate_mse(a_pde)
                pde_lib.append(a_pde)
                err_lib.append((a_err, a_w))
            print('Creating the ith pde, i=', i)
            print('a_pde.visualize():',a_pde.visualize())
            print('evaluate_aic:',a_err)
            self.eqs.append(a_pde)
            self.mses.append(a_err)

        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse) # 从小到大排序，提取出排序的index
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        self.mses, self.eqs = self.mses[0:num], self.eqs[0:num]
        
        print('Creating the original pdes in the pool 2...')
        for i in range(num*self.ratio): # 循环产生num个pde
            a_pde = PDE(depth, width, p_var)
            a_err, a_w = evaluate_mse(a_pde)
            pde_lib.append(a_pde)
            err_lib.append((a_err, a_w))
            while a_err < -100 or a_err == np.inf:  # MSE太小则直接去除，to avoid u d t
                print(a_err)
                a_pde = PDE(depth, width, p_var)
                a_err, a_w = evaluate_mse(a_pde)
                pde_lib.append(a_pde)
                err_lib.append((a_err, a_w))
            print('Creating the ith pde, i=', i)
            print('a_pde.visualize():',a_pde.visualize())
            print('evaluate_aic:',a_err)
            self.eqs2.append(a_pde)
            self.mses2.append(a_err)

        new_eqs2, new_mse2 = copy.deepcopy(self.eqs2), copy.deepcopy(self.mses2)
        sorted_indices2 = np.argsort(new_mse2) # 从小到大排序，提取出排序的index
        for i, ix in enumerate(sorted_indices2):
            self.mses2[i], self.eqs2[i] = new_mse2[ix], new_eqs2[ix]
        self.mses2, self.eqs2 = self.mses2[0:num], self.eqs2[0:num]
        
        print('Creating the original pdes in the pool 3...')
        for i in range(num*self.ratio): # 循环产生num个pde
            a_pde = PDE(depth, width, p_var)
            a_err, a_w = evaluate_mse(a_pde)
            pde_lib.append(a_pde)
            err_lib.append((a_err, a_w))
            while a_err < -100 or a_err == np.inf:  # MSE太小则直接去除，to avoid u d t
                print(a_err)
                a_pde = PDE(depth, width, p_var)
                a_err, a_w = evaluate_mse(a_pde)
                pde_lib.append(a_pde)
                err_lib.append((a_err, a_w))
            print('Creating the ith pde, i=', i)
            print('a_pde.visualize():',a_pde.visualize())
            print('evaluate_aic:',a_err)
            self.eqs3.append(a_pde)
            self.mses3.append(a_err)

        new_eqs3, new_mse3 = copy.deepcopy(self.eqs3), copy.deepcopy(self.mses3)
        sorted_indices3 = np.argsort(new_mse3) # 从小到大排序，提取出排序的index
        for i, ix in enumerate(sorted_indices3):
            self.mses3[i], self.eqs3[i] = new_mse3[ix], new_eqs3[ix]
        self.mses3, self.eqs3 = self.mses3[0:num], self.eqs3[0:num]

        # pdb.set_trace()

    def run(self, gen=100):
        for i in range(1, gen+1):
            print("###############################")
            self.cross_over(self.p_cro)
            self.change(self.p_mute, self.p_rep)
            best_eq, best_mse, index = self.the_best()
            
            self.change2()
            best_eq2, best_mse2, index2 = self.the_best2()
            self.change3()
            best_eq3, best_mse3, index3 = self.the_best3()
            new_eq3, new_mse3 = copy.deepcopy(self.eqs3[0]), copy.deepcopy(self.mses3[0])
            self.eqs.append(new_eq3)
            self.mses.append(new_mse3)
            new_eq2, new_mse2 = copy.deepcopy(self.eqs2[0]), copy.deepcopy(self.mses2[0])
            self.eqs.append(new_eq2)
            self.mses.append(new_mse2)
            
            if best_mse <= best_mse2: #123, 132, 312
                if best_mse <= best_mse3: #123, 132
                    print('{} generation best_aic & best Eq: {}, {}. [Pool 1]'.format(i, best_mse, best_eq.visualize()))
                    print('Best Concise Eq: {}'.format(best_eq.concise_visualize()))
                    print('{} generation repeat cross over {} times and mutation {} times'.
                        format(i, self.repeat_cross, self.repeat_change))
                    self.repeat_cross, self.repeat_change = 0, 0
                else: #312
                    print('{} generation best_aic & best Eq: {}, {}. [Pool 3]'.format(i, best_mse3, best_eq3.visualize()))
                    print('Best Concise Eq: {}'.format(best_eq3.concise_visualize()))
                    self.repeat_cross, self.repeat_change = 0, 0
                    #add best PDE in pool 3 to pool 2
                    self.eqs2.append(new_eq3)
                    self.mses2.append(new_mse3)
            else: #213, 231, 321
                if best_mse2 <= best_mse3: #213, 231
                    print('{} generation best_aic & best Eq: {}, {}. [Pool 2]'.format(i, best_mse2, best_eq2.visualize()))
                    print('Best Concise Eq: {}'.format(best_eq2.concise_visualize()))
                    self.repeat_cross, self.repeat_change = 0, 0 
                else: #321               
                    print('{} generation best_aic & best Eq: {}, {}. [Pool 3]'.format(i, best_mse3, best_eq3.visualize()))
                    print('Best Concise Eq: {}'.format(best_eq3.concise_visualize()))
                    self.repeat_cross, self.repeat_change = 0, 0
                    #add best PDE in pool 3 to pool 2
                    self.eqs2.append(new_eq3)
                    self.mses2.append(new_mse3)                  
                
            ############ Pool Check ###############
            if i % 20 == 0:
                #self.replenish()
                print("///////////////// Pool 1 /////////////////")
                for i in range(20):
                    print(i, 'th PDE: ', self.eqs[i].visualize(), ' AIC: ',self.mses[i])
                print("///////////////// Pool 2 /////////////////")
                for i in range(20):
                    print(i, 'th PDE: ', self.eqs2[i].visualize(), ' AIC: ',self.mses2[i])
                print("///////////////// Pool 3 /////////////////")
                for i in range(20):
                    print(i, 'th PDE: ', self.eqs3[i].visualize(), ' AIC: ',self.mses3[i])

    def the_best(self):
        argmin = np.argmin(self.mses)
        return self.eqs[argmin], self.mses[argmin], argmin
    def the_best2(self):
        argmin = np.argmin(self.mses2)
        return self.eqs2[argmin], self.mses2[argmin], argmin
    def the_best3(self):
        argmin = np.argmin(self.mses3)
        return self.eqs3[argmin], self.mses3[argmin], argmin

    def cross_over(self, percentage=0.5): # 比如一代有2n个样本，先用最好的n个样本交叉，产生了m个新的不重复的样本。则最终提取了2n+m个样本中最好的2n个。
        def cross_individual(pde1, pde2):
            new_pde1, new_pde2 = copy.deepcopy(pde1), copy.deepcopy(pde2)
            w1, w2 = len(pde1.elements), len(pde2.elements)
            ix1, ix2 = np.random.randint(w1), np.random.randint(w2)
            new_pde1.elements[ix1] = pde2.elements[ix2]
            new_pde2.elements[ix2] = pde1.elements[ix1]
            return new_pde1, new_pde2
        self.cross_num = 0
        # 一半的好样本保存，并在此基础上交叉生成一半新样本
        num_ix = int(self.num * percentage)

        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        copy_mses, copy_eqs = self.mses[0:num_ix], self.eqs[0:num_ix]  # top percentage samples

        new_eqs, new_mse = copy.deepcopy(copy_eqs), copy.deepcopy(copy_mses)
        reo_eqs, reo_mse = copy.deepcopy(copy_eqs), copy.deepcopy(copy_mses)
        random.shuffle(reo_mse)
        random.shuffle(reo_eqs)
        
        for a, b in zip(new_eqs, reo_eqs):
            new_a, new_b = cross_individual(a, b) # 在好样本的基础上交叉
            if new_a.visualize() in pde_lib:
                self.repeat_cross += 1
            else: # 前一半样本交叉产生了新的pde，则加入lib中，并且加入当前代的全部样本中
                a_err, a_w = evaluate_mse(new_a)
                pde_lib.append(new_a.visualize())
                err_lib.append((a_err, a_w))
                self.mses.append(a_err)
                self.eqs.append(new_a)
                self.cross_num += 1

            if new_b.visualize() in pde_lib:
                self.repeat_cross += 1
            else: # 前一半样本交叉产生了新的pde，则加入lib中，并且加入当前代的全部样本中
                b_err, b_w = evaluate_mse(new_b)
                pde_lib.append(new_b.visualize())
                err_lib.append((b_err, b_w))
                self.mses.append(b_err)
                self.eqs.append(new_b)
                self.cross_num += 1
                
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)#[0:self.num] # 对当前一代所有的样本和新增非重复样本，整体做一次排序，提取最优的本代样本数个样本。
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        
        self.mses = self.mses[:self.num+2]
        self.eqs = self.eqs[:self.num+2]


    def change(self, p_mute=0.05, p_rep=0.3):
        self.mut_num = 0
        self.rep_num = 0
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)

        for i in range(self.num):            
            # 保留最好的那部分eqs不change，只cross over.
            if i < 1: #保留最好的1个样本，不进行change
                continue
            new_eqs[i].mutate(p_mute)
            self.mut_num += 1

            replace_or_not = np.random.choice([False, True], p=([1 - p_rep, p_rep]))
            if replace_or_not:
                new_eqs[i].replace()
                self.rep_num += 1
            # 检测是否重复
            if new_eqs[i].visualize() in pde_lib:
                self.repeat_change += 1
            else:
                a_err, a_w = evaluate_mse(new_eqs[i])
                pde_lib.append(new_eqs[i].visualize())
                err_lib.append((a_err, a_w))
                self.mses.append(a_err)
                self.eqs.append(new_eqs[i])

        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)#[0:self.num] # 对当前一代所有的样本和新增非重复样本，整体做一次排序，提取最优的本代样本数个样本。
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
                
    def change2(self): #change method for pool 2 (100% mutate and replace the copy of the PDE in the pool, no crossover, rank and remain the top 20 PDEs)
        new_eqs, new_mse = copy.deepcopy(self.eqs2), copy.deepcopy(self.mses2)
        sorted_indices = np.argsort(new_mse)
        for i, ix in enumerate(sorted_indices):
            self.mses2[i], self.eqs2[i] = new_mse[ix], new_eqs[ix]
        new_eqs, new_mse = copy.deepcopy(self.eqs2), copy.deepcopy(self.mses2)

        for i in range(self.num):            
            new_eqs[i].mutate(1) #100% mutate to maximize randomness
            new_eqs[i].replace() #100% replace to maximize randomness

            # 检测是否重复
            if new_eqs[i].visualize() not in pde_lib:
                a_err, a_w = evaluate_mse(new_eqs[i])
                pde_lib.append(new_eqs[i].visualize())
                err_lib.append((a_err, a_w))
                self.mses2.append(a_err)
                self.eqs2.append(new_eqs[i])

        new_eqs, new_mse = copy.deepcopy(self.eqs2), copy.deepcopy(self.mses2)
        sorted_indices = np.argsort(new_mse)#[0:self.num] # 对当前一代所有的样本和新增非重复样本，整体做一次排序，提取最优的本代样本数个样本。
        for i, ix in enumerate(sorted_indices):
            self.mses2[i], self.eqs2[i] = new_mse[ix], new_eqs[ix]
        
        self.mses2 = self.mses2[:self.num+1]
        self.eqs2 = self.eqs2[:self.num+1]
            
    def change3(self): #change method for pool 3 (100% mutate and replace on the PDE in the pool)
        for i in range(self.num): 
            if self.eqs3[i].W == 0:
                self.eqs3[i] = PDE(4,10,0.5)
            else:         
                self.eqs3[i].mutate(1) #100% mutate to maximize randomness
                self.eqs3[i].replace() #100% replace to maximize randomness

            # 检测是否重复
            if self.eqs3[i].visualize() not in pde_lib:
                a_err, a_w = evaluate_mse(self.eqs3[i])
                pde_lib.append(self.eqs3[i].visualize())
                err_lib.append((a_err, a_w))
                self.mses3[i] = a_err

        new_eqs, new_mse = copy.deepcopy(self.eqs3), copy.deepcopy(self.mses3)
        sorted_indices = np.argsort(new_mse)#[0:self.num] # 对当前一代整体做一次排序
        for i, ix in enumerate(sorted_indices):
            self.mses3[i], self.eqs3[i] = new_mse[ix], new_eqs[ix]
        


if __name__ == '__main__':
    sys.stdout = Logger('notes.log', sys.stdout)
    sys.stderr = Logger('notes.log', sys.stderr)
    sga_num = 20
    sga_depth = 3#4
    sga_width = 8 #5 
    sga_p_var = 0.5
    sga_p_mute = 0.5#0.3
    sga_p_cro = 0.5#0.5
    sga_p_rep = 1
    sga_run = 100

    print('sga_num = ', sga_num)
    print('sga_depth = ', sga_depth)
    print('sga_width = ', sga_width)
    print('sga_p_var = ', sga_p_var)
    print('sga_p_mute = ', sga_p_mute)
    print('sga_p_rep = ', sga_p_rep)
    print('sga_p_cro = ', sga_p_cro)
    print('sga_run = ', sga_run)
    
    sga = SGA(num=sga_num, depth=sga_depth, width=sga_width, p_var=sga_p_var, p_rep=sga_p_rep, p_mute=sga_p_mute, p_cro=sga_p_cro)
    sga.run(sga_run)


