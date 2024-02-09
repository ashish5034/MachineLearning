from sklearn.datasets import load_iris

iris = load_iris()

print("Feature names of iris data set")
print(iris.feature_names)

print("Target names of iris data set")
print(iris.target_names)

print("Elements from iris data set")

# for i in range(len(iris.target)):
for i in range (10):
    print("ID: %d, Features %s, Label : %d" % (i,iris.data[i],iris.target[i]))

















