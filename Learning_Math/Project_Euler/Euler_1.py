 #attempted solution without modulo, didn't work.
 #def three_five(x):
 #    three_scaler = 0
 #    total_three = 0
 #    while three_scaler * 3 <= x:
 #        total_three += three_scaler * 3
 #        three_scaler += 1
 #    five_scaler = 0
 #    total_five = 0
 #    while five_scaler * 5 <= x:
 #        total_five += five_scaler * 5
 #        five_scaler += 1
 #    total = total_five + total_three
 #    return total
 #print(three_five(1000))
def three_five_sum(up_to):
    sum=0
    for i in range(up_to):
        if i%3==0 or i%5==0:
            sum+=i
    return sum
print(three_five_sum(1000))