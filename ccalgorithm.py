# encoding=utf8
from typing import Self, Tuple
import logging

import numpy as np

from niapy.task import Task
from niapy.algorithms.algorithm import AnalysisAlgorithm
from niapy.algorithms.algorithm import OptimizationAlgorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CooperativeCoevolution']


class CCTask(Task):
    def __init__(self, task:Task, inds:np.ndarray, *argrs, **kwargs):
        self.task = task
        self.inds = inds
        self.dimension = len(self.inds)
        self.lower = self.task.lower[self.inds]
        self.upper = self.task.upper[self.inds]
        self.range = self.task.range[self.inds]
        self.repair_function = self.task.repair_function
            
    def stopping_condition(self:Self) -> bool:
        return self.task.stopping_condition()

    def eval(self:Self, x:np.ndarray) -> float:
        xn = np.copy(self.task.lower)
        xn[self.inds] = x
        return self.task.eval(xn)


class CooperativeCoevolution(OptimizationAlgorithm):

    Name:list[str]=['CooperativeCoevolution', 'CC']

    def __init__(self:Self, decompozer:AnalysisAlgorithm, toptimizer:type[OptimizationAlgorithm], *args:list, **kwargs:dict[str, any]) -> None:
        kwargs.pop('population_size', None)
        super().__init__(1, *args, **kwargs)
        # set decompozer
        self.decompozer = decompozer
        # set optimizer
        self.toptimizer = toptimizer

    def set_parameters(self:Self, decompozer:AnalysisAlgorithm, toptimizer:type[OptimizationAlgorithm], **kwargs:dict[str, any]) -> None:
        kwargs.pop('population_size', None)
        super().set_parameters(**kwargs)
        # set decompozer
        self.decompozer = decompozer
        # set optimizer
        self.toptimizer = toptimizer

    def get_parameters(self:Self) -> dict[str, any]:
        params = super().get_parameters()
        params.pop('population_size', None)
        params.update({
            'decompozer': self.decompozer.Name[0],
            'optimizer': self.toptimizer.Name[0]
        })
        return params

    def _get_pop_size(self:Self, no_elems:int) -> int:
        r = no_elems * 10
        return r if r < 120 else 120

    def init_population(self:Self, task:Task) -> Tuple[np.ndarray, np.ndarray, dict[str, any]]:
        # get groups
        groups, seps = [], []
        for e in self.decompozer.run(task):
            if len(e) == 1: seps.append(e[0])
            else: groups.append(e)
        # init task for algorithms
        tasks = [CCTask(task, g) for g in groups]
        if len(seps) > 0: tasks.append(CCTask(task, seps))
        # init algorithms based on group sizes 
        pop, popf = [], []
        algs, algs_params = [], []
        best_x = []
        for t in tasks:
            no_pop = self._get_pop_size(t.dimension)
            a = self.toptimizer(population_size=no_pop)
            p, pf, d = a.init_population(t)
            algs.append(a)
            algs_params.append(d)
            pop.append(p)
            popf.append(pf)
        return pop, popf, {'groups': groups, 'seps': seps, 'tasks': tasks, 'algs': algs, 'algs_params': algs_params}

    def _get_x_best(self:Self, tasks:list[CCTask], xbs:list[np.ndarray]):
        x = np.copy(tasks[0].task.lower)
        for i, e in enumerate(xbs):
            x[tasks[i].inds] = e
        return x

    def get_best(self:Self, pop:list[np.ndarray], pop_f:list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        xbs, xbs_f = [], []
        for i, fs in enumerate(pop_f):
            xb_i = np.argmin(fs)
            xbs.append(np.copy(pop[i][xb_i]))
            xbs_f.append(fs[xb_i])
        return xbs, xbs_f

    def iteration_generator(self, task):
        iters = 0
        pop, fpop, params = self.init_population(task)
        xbs, fxbs = self.get_best(pop, fpop)
        xb = self._get_x_best(params['tasks'], xbs)
        fxb = task.eval(xb)
        if task.stopping_condition():
            yield xb, fxb
        while True:
            pop, fpop, xb, fxb, xbs, fxbs, params = self.run_iteration(task, pop, fpop, xb, fxb, xbs, fxbs, iters, **params)
            iters += 1
            yield xb, fxb

    def run_iteration(self:Self, task:Task, population:np.ndarray, population_fitness:np.ndarray, best_x:np.ndarray, best_fitness:np.ndarray, bests_x:list[np.ndarray], bests_fitness:list[float], iters:int, tasks:list[Task], algs:list[OptimizationAlgorithm], algs_params:list[dict[str, any]], groups:list[list[int]], seps:list[int], *args, **params:dict[str, any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, any]]:
        for i, t in enumerate(tasks):
            population[i], population_fitness[i], bests_x[i], bests_fitness[i], algs_params[i] = algs[i].run_iteration(t, population[i], population_fitness[i], bests_x[i], bests_fitness[i], iters, **algs_params[i])
            if bests_fitness[i] < best_fitness:
                best_x = np.copy(bests_x[i])
                best_fitness = bests_fitness[i]
        x = self._get_x_best(tasks, bests_x)
        fx = task.eval(x)
        if fx < best_fitness:
            best_fitness = fx
            best_x = np.copy(x)
        return population, population_fitness, best_x, best_fitness, bests_x, bests_fitness, {'groups': groups, 'seps': seps, 'tasks': tasks, 'algs': algs, 'algs_params': algs_params}

