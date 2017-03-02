from scipy import stats
import numpy as np

random_list = list()
for _ in range(30):
    random_list.append(np.sum(np.random.randint(2, size=3)))

print("----- Random list -----")
print("Mean: " + str(np.mean(random_list)))
print("Std: " + str(np.std(random_list)))
print("")

sample_4yo_array = np.array([0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
sample_3yo_array = np.array([2, 3, 2, 2, 1, 2, 1, 2, 0, 1, 0, 1, 1, 1, 2, 1, 0, 2, 0, 2, 2, 2, 1, 2, 2])

t, prob = stats.ttest_ind(sample_4yo_array, random_list, equal_var=True)
print("----- 4-years-old -----")
print("Mean: " + str(np.mean(sample_4yo_array)))
print("Std: " + str(np.std(sample_4yo_array)))
print("t = " + str(t))
print("prob = " + str(prob))
print("")

t, prob = stats.ttest_ind(sample_3yo_array, random_list, equal_var=True)
print("----- 3-years-old -----")
print("Mean: " + str(np.mean(sample_3yo_array)))
print("Std: " + str(np.std(sample_3yo_array)))
print("t = " + str(t))
print("prob = " + str(prob))
print("")





