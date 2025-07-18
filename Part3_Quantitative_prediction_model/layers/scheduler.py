import torch.optim as optim
from layers.scheduler_base import SchedulerBase

class Adam45(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam45, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 30e-5
        if epoch > 25:
            lr = 15e-5
        if epoch > 30:
            lr = 7.5e-5
        if epoch > 35:
            lr = 3e-5
        if epoch > 40:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class Adam45_1(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam45_1, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list = params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 30e-5
        if epoch > 5:
            lr = 15e-5
        if epoch > 10:
            lr = 7.5e-5
        if epoch > 15:
            lr = 3e-5
        if epoch > 20:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0005
        return self._cur_optimizer, self._lr

class Adam55(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam55, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self,net, epoch, epochs, **kwargs):
        lr = 30e-5
        if epoch > 25:
            lr = 15e-5
        if epoch > 35:
            lr = 7.5e-5
        if epoch > 45:
            lr = 3e-5
        if epoch > 50:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class SGD45(SchedulerBase):
    def __init__(self, params_list=None):
        super(SGD45, self).__init__()
        self._lr = 1e-2
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 1e-2
        if epoch > 5:
            lr = 1e-3
        if epoch > 10:
            lr = 1e-4
        if epoch > 15:
            lr = 15e-5
        if epoch > 30:
            lr = 7.5e-5
        if epoch > 35:
            lr = 3e-5
        if epoch > 40:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.SGD(net.parameters(), lr=lr,momentum=0.9, weight_decay=0.0001)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class FaceAdam(SchedulerBase):
    def __init__(self,params_list=None):
        super(FaceAdam, self).__init__()
        self._lr = 2e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 1e-4
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0005
        return self._cur_optimizer, self._lr

class CosineAdam():
    def __init__(self, params_list=None):
        super(CosineAdam, self).__init__()
        self._lr = 30e-5
        self._cur_optimizer = None
        self.params_list=params_list
        self._scheduler = None

    def schedule(self,net):
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=self._lr)
            self._scheduler = optim.lr_scheduler.CosineAnnealingLR(self._cur_optimizer, T_max=45, eta_min=1e-5)
        return self._cur_optimizer, self._scheduler

    def step_(self,epoch):
        if epoch < 45:
            self._scheduler.step()