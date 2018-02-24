def normalizeBin(value):
    if value > 255 : 
        return [1,1,1,1,1,1,1,1]

    r = value
    ret = [0,0,0,0,0,0,0,0]
    i = 0
    while r > 0:
        x = r % 2
        ret[i] = x
        r = int(r/2)
        i +=1
    return ret

print(normalizeBin(1))
print(normalizeBin(255))
print(normalizeBin(129))
print(normalizeBin(1000))