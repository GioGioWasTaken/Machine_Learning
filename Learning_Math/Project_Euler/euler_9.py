import matplotlib.pyplot as plt
def pythagorean_triplets(maximum):
    for a in range(1, maximum+1):
        for b in range(a, maximum+1):
            c = (a**2 + b**2)**0.5
            if c == int(c):
                if a + b + c == 1000:
                    return [a,b,c]
print(pythagorean_triplets(10000))