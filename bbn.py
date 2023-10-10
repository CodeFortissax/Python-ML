def is_prime(num):
    if num < 2:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False

    max_divisor = int(num ** 0.5) + 1
    for i in range(3, max_divisor, 2):
        if num % i == 0:
            return False

    return True

# Check if 9302023 is a prime number
number_to_check = 9302023
is_prime_number = is_prime(number_to_check)
print(f"Is {number_to_check} a prime number? {is_prime_number}")


s = 'clcoding'
print(s.center(5, '%'))