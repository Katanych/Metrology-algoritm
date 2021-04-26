'''Алгоритм обработки многократных равноточных имзерений'''
import numpy as np
from data_set_processing import DataSetProcessing


print("Алгоритм обработки многократных равноточных измерений\n")
print("Дано:")
num_data_sets = int(input("Количество измерений: ")) # к примеру 10
data_set = [10, 10.1, 10.2, 9.8, 9.9, 10, 9.9, 10.1, 10.8, 10]
# for index_data in range(num_data_sets):
#     data = input(f"{index_data+1} измерение: ")
#     data_set.append(float(data))
alpha = float(input("Доверительная вероятность: "))
first_experiences = DataSetProcessing(num_data_sets, data_set, alpha)

print("\nРешение:")
EXIST_OUTLIERS = True
while EXIST_OUTLIERS:
    mean_data_set = first_experiences.mean()
    print(f'Оценка измеряемой величины (СА):\nx = (1/n)*sum(xi) = {mean_data_set}')
    
    std_data_set = first_experiences.std()
    print(f'Среднеквадратичное отклонение (СКО):\nS = sqrt(sum((xi - x)^2)/(n-1)) = {std_data_set}')

    print("Проверка измерений на удовлетворение теоретическому значению критерия Граббса:")
    G1_max = first_experiences.g_right_tailed_test()
    print(f"Отклонение в максимальную сторону по критерию Граббса:\nG1 = abs(x_max - x) / S = {G1_max}")
    G2_min = first_experiences.g_left_tailed_test()
    print(f"Отклонение в минимальную сторону по критерию Граббса:\nG2 = abs(x_min - x) /  = {G2_min}")
    Gt = first_experiences.g_critical_value()
    print(f"Теоретическое значение Gt = {Gt}")

    if G1_max > Gt or G2_min > Gt:
        if G1_max > Gt:
            print(f"G1 - промах. Удаляем это измерение: {first_experiences.data_set.max()}")
            first_experiences.num_data_sets -= 1
            first_experiences.data_set = np.delete(first_experiences.data_set, np.where(first_experiences.data_set == first_experiences.data_set.max()))
        if G2_min > Gt:
            print(f"G2 - промах. Удаляем это измерение: {first_experiences.data_set.min()}")
            first_experiences.num_data_sets -= 1
            first_experiences.data_set = np.delete(first_experiences.data_set, np.where(first_experiences.data_set == first_experiences.data_set.min()))
        print(f"\nОбновленный набор измерений:\n{first_experiences.data_set}")
    else: 
        print("Промахов больше нет!\n")
        EXIST_OUTLIERS = False

print("Определение доверительных границ случайной погрешности.")
ts = first_experiences.t_student_test_value()
print(f"Значение коэффициента стьюдента: {ts}")
eps = first_experiences.confidence_bounds()
print("Границы случайной погрешности:")
print(f"eps = ts * S / sqrt(n) = {eps}")

print(f"\nОтвет: Rизм = ({first_experiences.mean()}+-{eps})Ом; Pд = {first_experiences.alpha}; n = {first_experiences.num_data_sets}.")
print("Погрешность округлите самостоятельно согласно правилам округления погрешности!")
print("Если первая значащая цифра 3 и меньше, округлить до двух значащих цифр!")
print("Если первая значащая цифра 4 и больше, округлить до одной значащей цифры!")
