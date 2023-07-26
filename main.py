# encoding=utf8
import sys
import timeit

import numpy as np
from numpy.random import rand

from cec2013lsgo.cec2013 import Benchmark

from niapy.task import Task
from niapy.problems import Problem

from niapy.algorithms.analysis import (
    RecursiveDifferentialGrouping, 
    ExtendedDifferentialGrouping
)
from niapy.algorithms.basic import (
    BareBonesFireworksAlgorithm,
    ParticleSwarmAlgorithm, 
    SineCosineAlgorithm,
    FireflyAlgorithm,
    HarmonySearch,
    BatAlgorithm
)


class CEC2013lsgoTask(Task):
    def __init__(self, no_fun:int, *args:list, **kwargs:dict)->None:
        if 1 < no_fun > 15: raise Exception('Function between 1 and 15!!!')
        bench = Benchmark()
        info = bench.get_info(no_fun)
        max_evals = 3e6
        fun_fitness = bench.get_function(no_fun)
        
        class CEC2013lsgoProblem(Problem):
            def __init__(self, *args:list, **kwargs:dict):
                kwargs.pop('dimension', None), kwargs.pop('lower', None), kwargs.pop('upper', None)
                super().__init__(dimension=info['dimension'], lower=info['lower'], upper=info['upper'], *args, **kwargs)
        
            def _evaluate(self, x):
                return fun_fitness(x)

        kwargs.pop('problem', None), kwargs.pop('optimization_type', None), kwargs.pop('lower', None), kwargs.pop('upper', None), kwargs.pop('dimension', None), kwargs.pop('max_evals', None)
        super().__init__(problem=CEC2013lsgoProblem(), max_evals=max_evals, *args, **kwargs)

    def get_mesures(self):
        r = [self.fitness_evals[0][1], self.fitness_evals[0][1], self.fitness_evals[0][1]]
        for e in self.fitness_evals:
            if e[0] > 120000: break
            else: r[0] = e[1]
        for e in self.fitness_evals:
            if e[0] > 600000: break
            else: r[1] = e[1]
        for e in self.fitness_evals:
            if e[0] > 3000000: break
            else: r[2] = e[1]
        return r


def run_algo(talgo:type, no_fun:int, seed:int, *args:list, **kwargs:dict)->None:
    for i in range(50):
        algo = talgo(seed=seed + i, **kwargs)
        task = CEC2013lsgoTask(no_fun=no_fun)
        start = timeit.default_timer()
        best = algo.run(task)
        stop = timeit.default_timer()
        with open('%s.cec2013lso.%d.csv' % (algo.Name[1], no_fun), 'a') as csvfile:
            f1, f2, f3 = task.get_mesures()
            csvfile.write('%d, %f, %f, %f, %f\n' % (seed + i, f1, f2, f3, stop - start))


def run_fa_cec2013(NP:int=100, seed:int=1, no_fun:int=1)->None:
    run_algo(FireflyAlgorithm, no_fun, 1, population_size=NP)


def run_hs_cec2013(NP:int=100, seed:int=1, no_fun:int=1)->None:
    run_algo(HarmonySearch, no_fun, 1, population_size=NP)


def run_ba_cec2013(NP:int=100, seed:int=1, no_fun:int=1)->None:
    run_algo(BatAlgorithm, no_fun, 1, population_size=NP)


def run_pso_cec2013(NP:int=100, seed:int=1, no_fun:int=1)->None:
    run_algo(ParticleSwarmAlgorithm, no_fun, 1, population_size=NP)


def run_sca_cec2013(NP:int=100, seed:int=1, no_fun:int=1)->None:
    run_algo(SineCosineAlgorithm, no_fun, 1, population_size=NP)


def run_bfwa_cec2013(NP:int=100, seed:int=1, no_fun:int=1)->None:
    run_algo(BareBonesFireworksAlgorithm, no_fun, 1, population_size=NP)


def run_rdg_cec2013(alpha:float=1e-12, NP:int=50, seed:int=1, no_fun:int=1)->None:
    algo = RecursiveDifferentialGrouping(seed=seed)
    for i in range(3):
        task = CEC2013lsgoTask(no_fun=no_fun)
        algo.set_parameters(n=NP, alpha=alpha)
        best = algo.run(task=task)
        print('groups: %s\nno. groups: %s\tno. evals: %d' % (best, len(best), task.evals))
        print(algo.Name[-1], ': ', algo.get_parameters())


def run_xdg_cec2013(alpha:float=1e-12, NP:int=50, seed:int=1, no_fun:int=1)->None:
    algo = ExtendedDifferentialGrouping(seed=seed)
    for i in range(3):
        task = CEC2013lsgoTask(no_fun=no_fun)
        # Get better mesurement for epsilon
        X = algo.rng.uniform(task.lower, task.upper, (NP, task.dimension))
        Xf = np.apply_along_axis(task.eval, 1, X)
        epsilon = alpha * np.min(Xf)
        # Set new epsilon
        algo.set_parameters(epsilon=epsilon)
        best = algo.run(task=task)
        print('groups: %s\nno. groups: %s\tno. evals: %d' % (best, len(best), task.evals))
        print(algo.Name[-1], ': ', algo.get_parameters())


def run_test_func(no_fun:int, max_evals:int=3e6)->None:
    if 1 < no_fun > 15: raise Exception('Function between 1 and 15!!!')
    bench = Benchmark()
    info = bench.get_info(no_fun)
    fun_fitness = bench.get_function(no_fun)
    start = timeit.default_timer()
    i = 0
    while i < max_evals:
        sol = info['lower'] + rand(info['dimension']) * (info['upper'] - info['lower'])
        fun_fitness(sol)
        i += 1
    end = timeit.default_timer()
    print('Time of execution for f%d for %d evals = %fs' % (no_fun, max_evals, end - start))


if __name__ == "__main__":
    arg_no_fun = int(sys.argv[1])
    #run_test_func(no_fun=arg_no_fun)
    #run_rdg_cec2013(no_fun=arg_no_fun)
    #run_xdg_cec2013(no_fun=arg_no_fun)
    run_bfwa_cec2013(no_fun=arg_no_fun)

