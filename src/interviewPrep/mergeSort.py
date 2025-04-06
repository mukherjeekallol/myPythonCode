from typing import List
l = [10,30,20,45,100,32,103,15,2]

def mergeList(l1:List,l2:List) -> List:
    ml = []
    print(l1,'  ',l2)

    while l1 and l2:
        if l1[0]<l2[0]:
            item = l1.pop(0)
        else:
            item = l2.pop(0)
        ml.append(item)
    
    if l1:
        ml.extend(l1)
    if l2:
        ml.extend(l2)
    
    return ml


def mergeSort(l):
    
    if len(l)<=1:
        return l
    else:
        mid = len(l)//2
        left=l[:mid]
        right=l[mid:]
        print(left,right)
        ms = mergeList(mergeSort(left),mergeSort(right))
        return ms

sl = mergeSort(l)
print(sl)
