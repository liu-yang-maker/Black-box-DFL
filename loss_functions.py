import torch
import numpy as np
import multiprocessing as mp
import math

def loss_modularity(mu, r, embeds, dist, bin_adj, mod, args):
    bin_adj_nodiag = bin_adj*(torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
    return (1./bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()

def loss_kcenter(mu, r, embeds, dist, bin_adj, obj, args):
    if obj == None:
        return torch.tensor(0).float()
    x = torch.softmax(dist*args.kcentertemp, 0).sum(dim=1)
    x = 2*(torch.sigmoid(4*x) - 0.5)
    if x.sum() > args.K:
        x = args.K*x/x.sum()
    loss = obj(x)
    return loss


def loss_modularity_simple(r, bin_adj, mod):
    bin_adj_nodiag = bin_adj*(torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
    # print(r.dtype)
    r, mod = r.double(), mod.double()

    return (1./bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()


def loss_kcenter_simple(dist, obj, args):
    if obj == None:
        return torch.tensor(0).float()
    x = torch.softmax(dist*args.kcentertemp, 0).sum(dim=1)
    x = 2*(torch.sigmoid(4*x) - 0.5)
    if x.sum() > args.K:
        x = args.K*x/x.sum()
    loss = obj(x)
    return loss



class loss_modularity_myloss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, dist, train_object, args):
        ctx.save_for_backward(dist)
        ctx.obj = train_object, args
        ctx.args = args
        return loss_kcenter_simple(dist, train_object, args)
    
    @staticmethod
    def backward(ctx, grad_output):
        print("we use the zeroth-order method...")

        mode = 1 # n: point

        # update r
        dist = ctx.saved_tensors
        train_object, args = ctx.obj, ctx.args
        N = dist.shape[0]
        repeat_time = int( math.sqrt(N))
        eps = 0.001

        estimation_r = []

        if mode == 1: # 1-point method
            print("we use the 1-point method...")

            obj_old = loss_kcenter_simple(dist, train_object, args)


            for _ in range(repeat_time):
                delta_r = np.random.random( (dist.shape[0], dist.shape[1]) )
                delta_r = delta_r/( np.linalg.norm(delta_r, ord=2) )

                r_permutation = dist + delta_r * eps
                obj_permutation = loss_kcenter_simple(r_permutation, train_object, args)
                
                estimation_r_item = torch.tensor( (obj_permutation-obj_old)*delta_r/eps )

                estimation_r.append( estimation_r_item )

        grad_input_r = torch.mean( torch.stack(estimation_r), dim=0 )
        print(grad_input_r.shape)

        return grad_input_r
    

def one_backward_loop(name, param):
        r, bin_adj, mod, eps, obj_old = param[0], param[1], param[2], param[3], param[4]
        delta_r = np.random.random( (r.shape[0], r.shape[1]) )
        delta_r = delta_r/( np.linalg.norm(delta_r, ord=2) )

        r_permutation = r + delta_r * eps
        obj_permutation = loss_modularity_simple(r_permutation, bin_adj, mod)
        
        estimation_r_item = torch.tensor( (obj_permutation-obj_old)*delta_r/eps )
        return {name: estimation_r_item}

class loss_modularity_myloss_parallel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, r, bin_adj, mod):
        ctx.save_for_backward(r, bin_adj, mod)
        return loss_modularity_simple(r, bin_adj, mod)
    
    @staticmethod
    def backward(ctx, grad_output):
        print("we use the zeroth-order method...")

        num_cores = int(mp.cpu_count())
        pool = mp.Pool(num_cores)


        mode = 1 # n: point

        # update r
        r, bin_adj, mod = ctx.saved_tensors
        N = r.shape[0]
        # repeat_time = int(N/2)
        repeat_time = num_cores
        print("repeat time: ", repeat_time)
        eps = 0.001

        estimation_r = []

        if mode == 1: # 1-point method
            print("we use the 1-point method...")

            obj_old = loss_modularity_simple(r, bin_adj, mod)
            
            param = [r.detach(), bin_adj, mod, eps, obj_old]

            param_dict = {}
            for _ in range(repeat_time):
                item_key = "task" + str(_)
                item_value = param
                param_dict[item_key] = item_value
            
            # param_dict = {'task1': list(range(10, 30000000)),
            #       'task2': list(range(30000000, 60000000)),
            #       'task3': list(range(60000000, 90000000)),
            #       'task4': list(range(90000000, 120000000)),
            #       'task5': list(range(120000000, 150000000)),
            #       'task6': list(range(150000000, 180000000)),
            #       'task7': list(range(180000000, 210000000)),
            #       'task8': list(range(210000000, 240000000))}

            results = [pool.apply_async(one_backward_loop, args=(name, param)) for name, param in param_dict.items()]
            print(results[0].get())
            assert 1==2
            results = [p.get() for p in results]
        print(len(results) )
        # assert 1==2

        grad_input_r = torch.mean( torch.stack(estimation_r), dim=0 )
        print(grad_input_r.shape)

        return grad_input_r, None, None
    




class loss_kcenter_myloss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, r, bin_adj, mod):
        ctx.save_for_backward(r, bin_adj, mod)
        return loss_kcenter_simple(r, bin_adj, mod)
    
    @staticmethod
    def backward(ctx, grad_output):
        print("we use the zeroth-order method...")

        mode = 1 # n: point

        # update r
        r, bin_adj, mod = ctx.saved_tensors
        N = r.shape[0]
        N = 20
        repeat_time = int(N/2)
        eps = 0.001

        estimation_r = []

        if mode == 1: # 1-point method
            print("we use the 1-point method...")

            obj_old = loss_modularity_simple(r, bin_adj, mod)


            for _ in range(repeat_time):
                delta_r = np.random.random( (r.shape[0], r.shape[1]) )
                delta_r = delta_r/( np.linalg.norm(delta_r, ord=2) )

                r_permutation = r + delta_r * eps
                obj_permutation = loss_modularity_simple(r_permutation, bin_adj, mod)
                
                estimation_r_item = torch.tensor( (obj_permutation-obj_old)*delta_r/eps )

                estimation_r.append( estimation_r_item )

        grad_input_r = torch.mean( torch.stack(estimation_r), dim=0 )
        print(grad_input_r.shape)

        return grad_input_r, None, None
    
