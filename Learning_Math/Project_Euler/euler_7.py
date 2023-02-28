def is_prime(num):
    if num < 2:
        return False
    elif num == 2:
        return True
    elif num % 2 == 0:
        return False
    else:
        for i in range(3, int(num**0.5)+1, 2): # we can jump every 2 numbers, because every even number (except 2) is not prime.
            if num % i == 0:
                return False
        return True
def count(until):
    numbers=[]
    i=1
    while len(numbers)<(until):
        if is_prime(i):
            numbers.append(i)
        i=i+1
    return numbers
list=count(10001)
print(len(list),list[10000])