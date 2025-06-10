# task.py - Окрема задача для демонстрації
"""
Задача: Створити функції для роботи з числами та їх тестування
"""

def factorial(n):
    """Обчислює факторіал числа n"""
    if n < 0:
        raise ValueError("Факторіал не визначений для від'ємних чисел")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Повертає n-те число Фібоначчі"""
    if n < 0:
        raise ValueError("n повинно бути невід'ємним")
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(n):
    """Перевіряє, чи є число n простим"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def gcd(a, b):
    """Знаходить найбільший спільний дільник двох чисел"""
    while b:
        a, b = b, a % b
    return abs(a)

def lcm(a, b):
    """Знаходить найменше спільне кратне двох чисел"""
    return abs(a * b) // gcd(a, b)

def prime_factors(n):
    """Розкладає число на прості множники"""
    if n <= 1:
        return []
    
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def test_functions():
    """Тестує всі функції"""
    print("🧮 Тестування математичних функцій\n")
    
    # Тест факторіала
    print("📊 Факторіал:")
    test_values = [0, 1, 5, 7]
    for val in test_values:
        print(f"  {val}! = {factorial(val)}")
    
    # Тест Фібоначчі
    print("\n🔢 Числа Фібоначчі:")
    for i in range(10):
        print(f"  F({i}) = {fibonacci(i)}")
    
    # Тест простих чисел
    print("\n🔍 Перевірка простих чисел:")
    test_numbers = [2, 3, 4, 17, 25, 29, 100]
    for num in test_numbers:
        result = "просте" if is_prime(num) else "складене"
        print(f"  {num} - {result}")
    
    # Тест НСД та НСК
    print("\n➗ НСД та НСК:")
    pairs = [(12, 18), (15, 25), (7, 13)]
    for a, b in pairs:
        print(f"  НСД({a}, {b}) = {gcd(a, b)}")
        print(f"  НСК({a}, {b}) = {lcm(a, b)}")
    
    # Тест розкладу на прості множники
    print("\n🔨 Розклад на прості множники:")
    numbers = [12, 60, 97, 100]
    for num in numbers:
        factors = prime_factors(num)
        print(f"  {num} = {' × '.join(map(str, factors))}")

def interactive_mode():
    """Інтерактивний режим для тестування функцій"""
    print("\n🎮 Інтерактивний режим")
    print("Доступні команди:")
    print("1 - Факторіал")
    print("2 - Фібоначчі") 
    print("3 - Перевірка на просте число")
    print("4 - НСД двох чисел")
    print("5 - НСК двох чисел")
    print("6 - Розклад на прості множники")
    print("0 - Вихід")
    
    while True:
        try:
            choice = input("\nВведіть номер команди: ").strip()
            
            if choice == '0':
                print("До побачення! 👋")
                break
            elif choice == '1':
                n = int(input("Введіть число для факторіала: "))
                print(f"Результат: {factorial(n)}")
            elif choice == '2':
                n = int(input("Введіть позицію числа Фібоначчі: "))
                print(f"Результат: {fibonacci(n)}")
            elif choice == '3':
                n = int(input("Введіть число для перевірки: "))
                result = "просте" if is_prime(n) else "складене"
                print(f"Число {n} - {result}")
            elif choice == '4':
                a = int(input("Введіть перше число: "))
                b = int(input("Введіть друге число: "))
                print(f"НСД({a}, {b}) = {gcd(a, b)}")
            elif choice == '5':
                a = int(input("Введіть перше число: "))
                b = int(input("Введіть друге число: "))
                print(f"НСК({a}, {b}) = {lcm(a, b)}")
            elif choice == '6':
                n = int(input("Введіть число для розкладу: "))
                factors = prime_factors(n)
                print(f"{n} = {' × '.join(map(str, factors))}")
            else:
                print("Невірна команда! Спробуйте ще раз.")
                
        except ValueError as e:
            print(f"Помилка: {e}")
        except Exception as e:
            print(f"Несподівана помилка: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("📚 МАТЕМАТИЧНІ ФУНКЦІЇ - ДЕМОНСТРАЦІЯ")
    print("=" * 50)
    
    # Запуск тестів
    test_functions()
    
    # Запуск інтерактивного режиму
    interactive_mode()