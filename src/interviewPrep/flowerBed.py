# from typing import List
# def canPlaceFlowers(flowerbed: List[int], n: int) -> bool:
#     p=False
#     f = [0] + flowerbed + [0]
#     l = len(f)
    
#     if l>2:
#         i = 1
#         while i<(l-1):
#             if sum(f[i-1:i+2])==0:
#                 n-=1
#                 f[i]=1
#                 i+=2
#             else:
#                 i+=1
#         if n<=0:
#             p=True
    
#     if l==1 and f[0]==0 and n==1:
#         p=True
#     if l==2 and sum(f)==0 and n==1:
#         p=True
    
#     return p

# f = [0,0,1,0,0]
# n =1

# print(canPlaceFlowers(f,n))

# s = 'abc'
# s = list(s)
# print(s)
# converted = ''.join(s)
# print(converted)

# for i in reversed(range(5)):
#     print(i)
# nums = [10,22,3,-1,0,33]
# n = len(nums)
# l = [1]*n
# r = [1]*n
# ans = [1]*n

# for i in range(1,n):
#     print("i={}, l[i-1]={}, nums[i-1]={}".format(i,l[i-1],nums[i-1]))
#     l[i]=l[i-1]*nums[i-1]
# print(l)

# from typing import List
# def increasingTriplet(nums: List[int]) -> bool:
#     i=0
#     flg=False
#     while i<len(nums)-2:       
#         flg=False
#         if (nums[i] < nums[i+1]) and (nums[i+1]<nums[i+2]):      
#             flg=True
#             break
#         i+=1
#     return flg
# print(increasingTriplet([0,4,2,1,0,-1,-3]))

from typing import List
def moveZeroes(nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    l = len(nums)
    i=0
    cnt=0
    while i < l and cnt<l-1:
        if nums[i]==0:
            nums[i:]=nums[i+1:]
            nums.append(0)
            cnt+=1
            print(nums)
        else:
            i+=1
    
l = [0,0,1]

moveZeroes(l)
        