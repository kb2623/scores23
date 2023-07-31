# encoding=utf8
import sys
import timeit

import numpy as np
from numpy.random import rand

from cec2013lsgo.cec2013 import Benchmark

from niapy.task import Task
from niapy.problems import Problem

from niapy.algorithms.analysis import (
    RecursiveDifferentialGroupingV3,
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

from ccalgorithm import CooperativeCoevolution


class CEC2013lsgoTask(Task):
    def __init__(self, no_fun:int, *args:list, **kwargs:dict) -> None:
        if 1 < no_fun > 15: raise Exception('Function between 1 and 15!!!')
        bench = Benchmark()
        info = bench.get_info(no_fun)
        max_evals = 3e6
        fun_fitness = bench.get_function(no_fun)
        
        class CEC2013lsgoProblem(Problem):
            def __init__(self, *args:list, **kwargs:dict) -> None:
                kwargs.pop('dimension', None), kwargs.pop('lower', None), kwargs.pop('upper', None)
                super().__init__(dimension=info['dimension'], lower=info['lower'], upper=info['upper'], *args, **kwargs)
        
            def _evaluate(self, x):
                return fun_fitness(x)

        kwargs.pop('problem', None), kwargs.pop('optimization_type', None), kwargs.pop('lower', None), kwargs.pop('upper', None), kwargs.pop('dimension', None), kwargs.pop('max_evals', None)
        super().__init__(problem=CEC2013lsgoProblem(), max_evals=max_evals, *args, **kwargs)

    def get_mesures(self) -> list[float]:
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


def run_algo(id:int, talgo:type, no_fun:int, *args:list, **kwargs:dict) -> None:
    algo = talgo(seed=id, **kwargs)
    task = CEC2013lsgoTask(no_fun=no_fun)
    start = timeit.default_timer()
    best = algo.run(task)
    stop = timeit.default_timer()
    with open('%s.cec2013lso.%d.csv' % (algo.Name[1], no_fun), 'a') as csvfile:
        f1, f2, f3 = task.get_mesures()
        csvfile.write('%d, %f, %f, %f, %f\n' % (id, f1, f2, f3, stop - start))


def run_algo_50(talgo:type, no_fun:int, seed:int, *args:list, **kwargs:dict) -> None:
    for i in range(50): run_algo(seed + i, talgo, no_fun)
        

def run_fa_cec2013(NP:int = 100, seed:int = 1, no_fun:int = 1) -> None:
    run_algo_50(FireflyAlgorithm, no_fun, 1, population_size=NP)


def run_hs_cec2013(NP:int = 100, seed:int = 1, no_fun:int = 1) -> None:
    run_algo_50(HarmonySearch, no_fun, 1, population_size=NP)


def run_ba_cec2013(NP:int = 100, seed:int = 1, no_fun:int = 1) -> None:
    run_algo_50(BatAlgorithm, no_fun, 1, population_size=NP)


def run_pso_cec2013(NP:int = 100, seed:int = 1, no_fun:int = 1) -> None:
    run_algo_50(ParticleSwarmAlgorithm, no_fun, 1, population_size=NP)


def run_sca_cec2013(NP:int = 100, seed:int = 1, no_fun:int = 1) -> None:
    run_algo_50(SineCosineAlgorithm, no_fun, 1, population_size=NP)


def run_bbfwa_cec2013(NP:int = 100, seed:int = 1, no_fun:int = 1) -> None:
    run_algo_50(BareBonesFireworksAlgorithm, no_fun, 1, population_size=NP)


def no_seps(a:list) -> int:
    s = 0
    for e in a:
        if len(e) > 1: continue
        s += 1
    return s


def no_groups(a:list) -> int:
    s = 0
    for e in a:
        if len(e) == 1: continue
        s += 1
    return s


def run_rdg_cec2013(alpha:float = 1e-12, NP:int = 50, seed:int = 1, no_fun:int = 1) -> None:
    algo = RecursiveDifferentialGroupingV3(seed=seed)
    task = CEC2013lsgoTask(no_fun=no_fun)
    algo.set_parameters(n=NP, alpha=alpha)
    best = algo.run(task=task)
    print('groups: %s\nno. groups: %d\tno. seps: %d\nno. evals: %d' % (best, no_groups(best), no_seps(best), task.evals))
    print(algo.Name[-1], ': ', algo.get_parameters())


def run_xdg_cec2013(alpha:float = 1e-12, NP:int = 50, seed:int = 1, no_fun:int = 1) -> None:
    algo = ExtendedDifferentialGrouping(seed=seed)
    task = CEC2013lsgoTask(no_fun=no_fun)
    # Get better mesurement for epsilon
    X = algo.rng.uniform(task.lower, task.upper, (NP, task.dimension))
    Xf = np.apply_along_axis(task.eval, 1, X)
    epsilon = alpha * np.min(Xf)
    # Set new epsilon
    algo.set_parameters(epsilon=epsilon)
    best = algo.run(task=task)
    print('groups: %s\nno. groups: %s\nno. seps: %s\tno. evals: %d' % (best, no_groups(best), no_seps(best), task.evals))
    print(algo.Name[-1], ': ', algo.get_parameters())


def run_test_func(no_fun:int, max_evals:int = 3e6) -> None:
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


def run_cc_cec2013(no_fun:int, seed:int = 1) -> None:
    algo = CooperativeCoevolution(RecursiveDifferentialGroupingV3(), BareBonesFireworksAlgorithm, seed=seed)
    #algo = CooperativeCoevolution(RecursiveDifferentialGroupingV3(), SineCosineAlgorithm, seed=seed)
    # create a test cec2013lsgo
    task = CEC2013lsgoTask(no_fun=no_fun)
    # start optimization of the task
    start = timeit.default_timer()
    res = algo.run(task)
    stop = timeit.default_timer()
    # TODO save results
    with open('%s.%s.cec2013lso.%d.csv' % (algo.decompozer.Name[1], algo.toptimizer.Name[1], no_fun), 'a') as csvfile:
        f1, f2, f3 = task.get_mesures()
        csvfile.write('%d, %f, %f, %f, %f\n' % (seed, f1, f2, f3, stop - start))


if __name__ == "__main__":
    arg_no_fun = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    arg_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    #run_test_func(no_fun=arg_no_fun)
    #run_rdg_cec2013(no_fun=arg_no_fun)
    #run_xdg_cec2013(no_fun=arg_no_fun)
    #run_bbfwa_cec2013(no_fun=arg_no_fun)
    run_cc_cec2013(arg_no_fun, arg_seed)

