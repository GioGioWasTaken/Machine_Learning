def fibonacci(upto):
    fibo_list=[]
    fibo=1
    temp_fibo=0
    final_fibo=0
    while len(str(final_fibo))<upto:
        final_fibo=fibo+temp_fibo
        fibo = temp_fibo
        temp_fibo=final_fibo
        fibo_list.append(final_fibo)
    return fibo_list
fibo_list=fibonacci(1000)
print(fibo_list,len(fibo_list))