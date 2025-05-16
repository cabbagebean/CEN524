import math  # Importing math library for potential future enhancements

def geometric_series_sum(a, r, n):
    """
    Computes the sum of a geometric series.

    Parameters:
    a - First term of the series (float)
    r - Common ratio (float)
    n - Number of terms (int) or None for infinite series

    Returns:
    The sum of the geometric series or a message if the series does not converge.
    """
    if n is None:  # Case for infinite series
        if abs(r) < 1:
            return a / (1 - r)  # Formula for infinite sum when |r| < 1
        else:
            return "Infinite sum does not converge"  # No valid sum if |r| >= 1

    # Case for finite series
    if abs(r) < 1:
        return a * (1 - r**n) / (1 - r)  # Formula for |r| < 1
    else:
        return a * (r**n - 1) / (r - 1)  # Formula for |r| > 1

# User input
a = float(input("Enter the first term (a): "))
r = float(input("Enter the common ratio (r): "))
n_input = input("Enter the number of terms (n) or type 'inf' for infinite sum: ")

# Handling infinite series case
n = None if n_input.lower() == 'inf' else int(n_input)

# Compute the result
result = geometric_series_sum(a, r, n)

# Output the result
print("Sum of the geometric series:", result)
