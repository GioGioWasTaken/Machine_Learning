def fibonacci(upto):
    fibo=1
    temp_fibo=2
    final_fibo=0
    print(fibo)
    print(temp_fibo)
    sum=2 # the sum starts from 2 because temp_fibo starts from 2
    while final_fibo<upto:
        final_fibo=fibo+temp_fibo
        print(final_fibo)
        fibo = temp_fibo
        temp_fibo=final_fibo
        if final_fibo%2==0:
            sum=sum+final_fibo
    return sum
sum=fibonacci(4000000)
print(f"I'm the sum: {sum}")