def is_divisible(num,upto):
    for i in range(2,upto+1):
        if num % i != 0:
            return False
    return True
def smallest_dividor(upto):
    num=upto
    while is_divisible(num,upto)==False:
        num=num+upto
    return num
print(smallest_dividor(20))