import numpy as np
# myList = [1,23,4,5,6]
# arr = np.array(myList)#list to array
# print(arr)
# print(type(arr))#ndArray

#multidimentional array
# arr = np.array([[1,3, 4], [4, 6, 3]])
# print(arr)
# print(arr.shape)#gives dimension of array
# print(arr.reshape(3,2))#reshapes tthe array based on dimensions

# print(arr[1, 2]) #gives the value of at the present index

# arr = [[1, 2, 3, 4], [4, 56, 7, 3], [9, 3, 5,7]]
# print(arr)
# arr = arr[0:][1:]
# print(arr)
# print(arr[:][2:])

# arr = np.arange(0, 9)#this is create a array of all elements in between o an 9
# print(arr)
# arr = np.arange(0, 9, 3)#3 argument is increment
# print(arr)

# arr = np.linspace(1, 10, 50)
# #creates the array whose elemenst fall between 1 and 10 and the 50 elements will be present which will be equally spaced
# print(arr)

# arr = np.ones((6, 2), dtype= int)
# print(arr)
# #This will create the array of dimension (6, 2) of int where only ones will be present as elements

arr = np.random.rand(3, 5)
print(arr)
#create the array of dimension (3,5) whose elements will be spaced randomly