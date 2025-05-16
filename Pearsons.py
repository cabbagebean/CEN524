import math

def pearson_correlation(X, Y):
    n = len(X)
    mean_x = sum(X) / n
    mean_y = sum(Y) / n

    num = sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(n))
    denom_x = sum((X[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((Y[i] - mean_y) ** 2 for i in range(n))

    if denom_x == 0 or denom_y == 0:
        return 0  # Avoid division by zero

    return num / math.sqrt(denom_x * denom_y)

# User input
n = int(input("Enter the number of values: "))
X = [float(input(f"Enter X[{i+1}]: ")) for i in range(n)]
Y = [float(input(f"Enter Y[{i+1}]: ")) for i in range(n)]

print("Pearson Correlation Coefficient:", pearson_correlation(X, Y))
