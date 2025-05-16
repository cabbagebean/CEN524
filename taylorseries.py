import math

def taylor_series_sum(F, a, x, n):
    sum_result = 0
    for i in range(n):
        term = (F[i] / math.factorial(i+1)) * ((x - a) ** (i+1))
        sum_result += term
    return sum_result

# User input
n = int(input("Enter the number of terms: "))
F = [float(input(f"Enter f^{i}(a): ")) for i in range(n)]
a = float(input("Enter the value of a: "))
x = float(input("Enter the value of x: "))

print("Taylor Series Sum:", taylor_series_sum(F, a, x, n))
