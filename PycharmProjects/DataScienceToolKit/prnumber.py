# Function to check if a number is prime
R = range(1, 1000001)


def is_prime(number):
    if number <= 1:
        return False
    if number <= 3:
        return True
    if number % 2 == 0 or number % 3 == 0:
        return False
    i = 5
    while i * i <= number:
        if number % i == 0 or number % (i + 2) == 0:
            return False
        i += 6
    return True

def count_primes():
    count = 0
    for num in R:
        if is_prime(num):
            count += 1
    return count


total_primes = count_primes()
print("Total prime numbers between 1 and 1,000,000:", total_primes)


# Function to get the first 5 numbers that the input number is divisible by
def first_divisible_numbers(number):
    divisors = []
    divisor = 2  # Start checking from 2
    while len(divisors) < 5 and divisor <= number:
        if number % divisor == 0:
            divisors.append(divisor)
        divisor += 1
    return divisors


# Main program
if __name__ == "__main__":
    while True:
        try:

            # Ask the user for input and check if it's prime
            user_input = int(input("Enter a number between 1 and 1,000,000: "))

            if 1 <= user_input <= 1000000:
                if is_prime(user_input):
                    print(f"{user_input} is a prime number. Impressive!")
                else:
                    print(f"{user_input} is not a prime number.")
                    divisors = first_divisible_numbers(user_input)
                    print(f"The first 5 numbers {user_input} is divisible by: {divisors}")
            else:
                print("Input is out of range (1 to 1451,000,000).")

        except ValueError:
            print("Invalid input. Please enter a valid integer.")

        # Ask if the user wants to perform another action
        another_number = input("Do you want to try another number? (yes/no): ").strip().lower()
        if another_number == 'yes':
            swali = input("Please get this right ama ukuwe blocked. Sawa? (sawa/zii): ").strip().lower()
            if swali != 'sawa':
                print("Nkt! Tanye wewe. BLOCKED!!")
                break

        if another_number != 'yes':
            print("Thank you. Goodbye!")
            break
