from sklearn import tree

def main():
    
    print("Ball classification case study")
    BallFeatures = [[35,"Rough"],[47,"Rough"],[90,"Smooth"],[48,"Rough"],[90,"Smooth"],[35,"Rough"],[92,"Smooth"],[35,"Rough"],[35,"Rough"],[35,"Rough"],[96,"Smooth"],[43,"Rough"],[110,"Smooth"],[35,"Rough"],[95,"Smooth"]]
    
    Labels = ["Teenis","Teenis","Cricket","Teenis","Cricket","Teenis","Cricket","Teenis","Teenis","Teenis","Cricket","Teenis","Cricket","Teenis","Cricket"]
    
    obj = tree.DecisionTreeClassifier()                         #descide the algorithm
    
    obj = obj.fit(BallFeatures,Labels)                              #train the model
    print(obj.predict([[36,"Rough"],[91,"Smooth"]]))        #test the model
    
if __name__ == "__main__":
    main()