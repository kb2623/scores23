# encoding=utf8
import sys
from typing import Self, Tuple, Union
import logging

import numpy as np

from niapy.task import Task
from niapy.algorithms.algorithm import AnalysisAlgorithm
from niapy.algorithms.algorithm import OptimizationAlgorithm
from niapy.algorithms.algorithm import CoEvolutionOptimizationAlgorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CooperativeCoevolutionV1']


class CooperativeCoevolutionV1(OptimizationAlgorithm):

    Name:list[str]=['CooperativeCoevolutionV1', 'CCv1']

    def __init__(self:Self, decompozer:AnalysisAlgorithm, toptimizer:type[CoEvolutionOptimizationAlgorithm], *args:list, **kwargs:dict[str, any]) -> None:
        super().__init__(decompozer=decompozer, toptimizer=toptimizer, *args, **kwargs)

    def set_parameters(self:Self, decompozer:AnalysisAlgorithm, toptimizer:type[OptimizationAlgorithm], **kwargs:dict[str, any]) -> None:
        kwargs.pop('population_size', None)
        super().set_parameters(**kwargs)
        # set decompozer
        self.decompozer = decompozer
        # set optimizer
        self.toptimizer = toptimizer
        # set optimizer empty parametes
        self.toptimizer_params = {}

    def get_parameters(self:Self) -> dict[str, any]:
        params = super().get_parameters()
        params.update({
            'decompozer': self.decompozer.Name[0],
            'optimizer': self.toptimizer.Name[0]
        })
        return params

    def set_decomposer_parameters(self:Self, *args:list, **kwargs:dict[str, any]) -> None:
        self.decompozer.set_parameters(*args, **kwargs)

    def get_decomposer_parameters(self:Self, *args:list, **kwargs:dict[str, any]) -> dict[str, any]:
        return self.decompozer.get_parameters()

    def set_optimizer_parameters(self:Self, *args:list, **kwargs:dict[str, any]) -> None:
        self.toptimizer_params = kwargs

    def get_optimizer_parameters(self:Self, *args:list, **kwargs:dict[str, any]) -> dict[str, any]:
        return self.toptimizer_params

    def init_population(self:Self, task:Task) -> Tuple[np.ndarray, np.ndarray, dict[str, any]]:
        # get groups
        G = self.decompozer.run(task)
        groups, seps = [], []
        for e in G:
            if len(e) == 1: seps.append(e[0])
            else: groups.append(e)
        # init task for algorithms
        if len(seps) > 0: groups.append(seps)
        # init algorithms based on group sizes 
        pop, popf = [], []
        algs, algs_params = [], []
        for g in groups:
            a = self.toptimizer(seed=self.integers(sys.maxsize))
            a.set_parameters(**self.toptimizer_params)
            p, pf, d = a.init_population(task, groups)
            algs.append(a)
            algs_params.append(d)
            pop.extend(p)
            popf.extend(pf)
        # sort and select best
        xf_si = np.argsort(popf)
        s = int(len(pop) / len(groups))
        if len(pop) > 1: pop, popf = pop[xf_si[:s]], popf[xf_si[:s]]
        return pop, popf, {'groups': groups, 'algs': algs, 'algs_params': algs_params}

    def run_iteration(self:Self, task:Task, population:np.ndarray, population_fitness:np.ndarray, best_x:np.ndarray, best_fitness:np.ndarray, iters:int, groups:Union[list[int], set[int], np.ndarray], algs:list[OptimizationAlgorithm], algs_params:list[dict[str, any]], *args, **params:dict[str, any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, any]]:
        for g in groups:
            pop, fpop, xb, fxb, params = self.run_iteration(task=task, population=pop, population_fitness=fpop, best_x=xb, best_fitness=fxb, iters=iters, groups=groups, *args, **params)
            # TODO if need be
        return population, population_fitness, best_x, best_fitness, bests_x, bests_fitness, {'groups': groups, 'seps': seps, 'tasks': tasks, 'algs': algs, 'algs_params': algs_params}

