l = [1,10,15,34,55,77,99,123]
target = 10

def binarySearch(l,target):
    start = 0
    end = len(l) - 1
    found = False
    while start <= end:
        mid = (start+end)//2
        if l[mid]==target:
            found = True
            break
        elif l[mid]>target:
            end = mid
        else:
            start=mid+1
    return(found)
        
if binarySearch(l,target):
    print("Found")
else:
    print("Not found")

    


