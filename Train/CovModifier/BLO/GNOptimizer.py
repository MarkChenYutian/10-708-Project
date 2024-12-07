import pypose as pp
from pypose.optim.optimizer import *    #type: ignore
from pypose.optim.optimizer import _Optimizer


class GaussNewton2(_Optimizer):
    def __init__(self, model, solver=None, kernel=None, corrector=None, vectorize=True):
        super().__init__(model.parameters(), defaults={})
        self.jackwargs = {'vectorize': vectorize}
        self.solver = PINV() if solver is None else solver
        if kernel is not None:
            kernel = [kernel] if not isinstance(kernel, (tuple, list)) else kernel
            kernel = [k if k is not None else Trivial() for k in kernel]
            self.corrector = [FastTriggs(k) for k in kernel] if corrector is None else corrector
        else:
            self.corrector = [Trivial()] if corrector is None else corrector
        self.corrector = [self.corrector] if not isinstance(self.corrector, (tuple, list)) else self.corrector
        self.corrector = [c if c is not None else Trivial() for c in self.corrector]
        self.model = RobustModel(model, kernel)

    @staticmethod
    def grad_update_parameter(params, step, updates):
        r'''
        params will be updated by calling this function
        '''
        steps = step.split([p.numel() for p in params if p.requires_grad])
        p: torch.Tensor
        
        updates.append([])
        for idx, (p, d) in enumerate(zip(params, steps)):
            if not p.requires_grad: continue
            # params[idx] = pp.Parameter(pp.LieTensor(d.view(p.shape)[..., :6], ltype=pp.se3_type).Exp() * p)
            # breakpoint()
            params[idx].data = (pp.LieTensor(d.view(p.shape)[..., :6], ltype=pp.se3_type).Exp() * p)
            updates[-1].append(pp.LieTensor(d.view(p.shape)[..., :6], ltype=pp.se3_type).Exp())

    def update(self, input, weight, target=None):
        updates = []
        
        for pg in self.param_groups:
            R = list(self.model(input, target))
            J = modjac(self.model, input=(input, target), flatten=False, vectorize=True, create_graph=False)
            
            params = dict(self.model.named_parameters())
            params_values = tuple(params.values())
            
            J = [self.model.flatten_row_jacobian(Jr, params_values) for Jr in J]
            
            for i in range(len(R)):
                R[i], J[i] = self.corrector[0](R = R[i], J = J[i])
            
            R, weight_mod, J = self.model.normalize_RWJ(R, weight, J)
            assert weight_mod is not None
            
            A, b =  weight_mod @ J, -weight_mod @ R
            D = self.solver(A = A, b = b.view(-1, 1))
            
            self.last = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            
            self.grad_update_parameter(params=pg['params'], step=D, updates=updates)
            self.loss = self.model.loss(input, target)
        
        return self.loss, updates
