# encoding=utf8
from niapy.algorithms.analysis import RecursiveDifferentialGroupingV2
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class ABCTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = RecursiveDifferentialGroupingV2

    def test_custom(self):
        abc_custom = self.algo(population_size=10, limit=2, seed=self.seed)
        abc_customc = self.algo(population_size=10, limit=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, abc_custom, abc_customc, MyProblem())

    def test_griewank(self):
        abc_griewank = self.algo(population_size=10, seed=self.seed)
        abc_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, abc_griewank, abc_griewankc)
