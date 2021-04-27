'''Модуль класса для обработки измерений'''
import numpy as np
from scipy.stats import t
from scipy import stats


class DataSetProcessing():
    '''Класс обработки измерений
    Определение полей класса:
    num_data_sets - количество измерений;
    data_set - набор данных измерений
    alpha - доверительная вероятность.

    '''
    def __init__(self, num_data_sets: int, data_set, alpha=0.95):
        '''Конструктор класса.
        Инициализация полей передаваемыми элементами.

        '''
        self.num_data_sets = num_data_sets
        self.data_set = np.array(data_set)
        self.alpha = float(alpha)
    
    def mean(self):
        '''Оценка измеряемой величины
        Определяется как среднее арифметическое результатов измерений.
        
        '''
        return np.mean(self.data_set)
    
    def std(self):
        '''Среднее квадратическое отклонение каждого измерения'''
        return np.std(self.data_set, ddof=1)

    def g_left_tailed_test(self):
        '''Критерий Граббса для отклонения в минимальную сторону'''
        return abs(self.data_set.min() - self.mean()) / self.std()

    def g_right_tailed_test(self):
        '''Критерий Граббса для отклонения в максимальную сторону'''
        return abs(self.data_set.max() - self.mean()) / self.std()

    def g_critical_value(self):
        '''Теоретическое значение критерия Граббса G'''
        pd = 1 - self.alpha
        t_dist = stats.t.ppf(1 - pd / (2 * self.num_data_sets), self.num_data_sets - 2)
        numerator = (self.num_data_sets - 1) * np.sqrt(np.square(t_dist))
        denominator = np.sqrt(self.num_data_sets) * np.sqrt(self.num_data_sets - 2 + np.square(t_dist))
        return numerator / denominator

    def t_student_test_value(self):
        '''Получение значения коэффициента стьюдента'''
        pd = 1 - self.alpha
        return t.ppf(1-(pd/2), self.num_data_sets-1)
    
    def confidence_bounds(self):
        '''Определение доверительных границ случайной погрешности'''
        return self.t_student_test_value() * self.std() / self.num_data_sets**(0.5)
