def sum_squares(x):
    return sum([i*i for i in range(1,x+1)])
sum_squares=sum_squares(100)

def squared_sum(x):
    total=0
    for i in range(1,x+1):
        total+=i
    return total*total
squared_nums=squared_sum(100)
print(squared_nums-sum_squares)