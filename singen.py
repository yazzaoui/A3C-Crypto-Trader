import math
j = 0
while j < 8 :
    i = 0
    with open('training/training_'+str(j)+'.data','w') as f:
        while i < (200 + j * 77) : 
            sign = (j% 2) * 2 - 1 
            amp = (100  )
            dec = 150 
            freq = i / (5 )
            val =  ( sign * amp * math.sin(  freq + j/2 ) + dec ) / 5
            f.write("aa aa "   + str(val) + " 0 0 0 0\n")
            i +=1
    j += 1        