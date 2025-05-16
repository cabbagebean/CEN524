def newton_reciprocal(N, x0=0.1, tol=1e-6, max_iter=10):
    for i in range(max_iter):
        x_next = x0 * (2 - x0 * N)
        print(f"Iteration {i + 1}: {x_next}")
        if abs(x_next - x0) < tol:
            break
        x0 = x_next
    return x_next

# User input
N = float(input("Enter the number N: "))
x0 = float(input("Enter the initial guess (default 0.1): ") or 0.1)

print("Approximate reciprocal:", newton_reciprocal(N, x0))
