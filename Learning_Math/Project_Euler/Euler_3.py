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

def factor(is_prime,upto):
    num = int(upto**0.5) # if a number has a factor above the square root, it also has the other part of the factor below it.
    while upto % num!=0:
        num-=1
    if is_prime(num):
        if upto / num>num:
            num=upto / num
        return num
    else:
        print(f"Non-prime factor detected: {num}")
        return factor(is_prime, num)
print(factor(is_prime,600851475143  ))
