def factorial(n):
    if n == 0 or n == 1:
        return 1
    fact = 1
    for i in range(2, n + 1):
        fact *= i
    return fact

# User input
num = int(input("Enter a number to find its factorial: "))

print(f"Factorial of {num} is:", factorial(num))
