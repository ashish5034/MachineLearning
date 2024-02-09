from sklearn import tree
# rough =1 
# smooth = 0
def main():
    
    print("Ball classification case study")
    BallFeatures = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    
    Labels = ["Teenis","Teenis","Cricket","Teenis","Cricket","Teenis","Cricket","Teenis","Teenis","Teenis","Cricket","Teenis","Cricket","Teenis","Cricket"]
    
    obj = tree.DecisionTreeClassifier() #descide the algorithm
    
    obj = obj.fit(BallFeatures,Labels)  #train the model
    
    print(obj.predict([[36,1],[91,0]])) #test the model
    
if __name__ == "__main__":
    main()