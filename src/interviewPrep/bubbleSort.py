l = [10,30,20,45,100,32,103,15,2]

for i in range(len(l)):
    print(f"i={i}, l={l}")
    for j in range(i+1,len(l)):
        print(f"i,j={i},{j}, l={l}")
        if l[i] > l[j]:
            l[j],l[i]=l[i],l[j]
            print(f"i,j={i},{j}, l={l}")

print(l)

