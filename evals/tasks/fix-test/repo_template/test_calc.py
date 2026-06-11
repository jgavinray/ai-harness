from calc import add, multiply

assert add(1, 2) == 3, f"add(1,2) should be 3, got {add(1, 2)}"
assert add(-1, 1) == 0
assert multiply(3, 4) == 12
print("all tests passed")
