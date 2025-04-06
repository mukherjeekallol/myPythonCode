inputList = [1,2,[4,5,6],[11,[23,35,[10,12]]]]

def flattenList(inputList):
    flatList=[]
    for item in inputList:
        if type(item)==list:
            flatList.extend(flattenList(item))
        else:
            flatList.append(item)
    
    return flatList

print(flattenList(inputList))
