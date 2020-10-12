import numpy as np 

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# print(x)

# 配列を作る
my_list1 = [i for i in range(1,6)]
my_list2 = [i for i in range(6,11)]
#print(my_list1 + my_list2)
my_lists = [my_list1, my_list2]
#print(my_lists)
my_array = np.array(my_lists)
#print(my_array)
#print(my_array.shape)

# 配列の生成
x = np.arange(10)
#print(x)
x = np.arange(1,10).reshape(3,3)
y = np.arange(1,10).reshape(3,3)

print(x*y)