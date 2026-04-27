# NUMERIC TYPES
# -------------


# Integer (int) - whole numbers
integer_example = 42
print(f"Integer: {integer_example}, Type: {type(integer_example)}")

# Float - decimal numbers
float_example = 3.14159
print(f"Float: {float_example}, Type: {type(float_example)}")

# Complex - numbers with real and imaginary parts
complex_example = 3 + 4j
print(f"Complex: {complex_example}, Type: {type(complex_example)}")


# SEQUENCE TYPES
# -------------

# String (str) - sequence of characters
string_example = "Hello, Python!"
print(f"String: {string_example}, Type: {type(string_example)}")

# List - ordered, mutable collection
list_example = [1, 2, 3, "four", 5.0]
print(f"List: {list_example}, Type: {type(list_example)}")

# Tuple - ordered, immutable collection
tuple_example = (1, 2, 3, "four", 5.0)
print(f"Tuple: {tuple_example}, Type: {type(tuple_example)}")

# Range - sequence of numbers
range_example = range(5)  # 0, 1, 2, 3, 4
print(f"Range: {range_example}, Type: {type(range_example)}")


# MAPPING TYPE
# ------------

# Dictionary (dict) - key-value pairs
dict_example = {"name": "Python", "version": 3.10, "is_fun": True}
print(f"Dictionary: {dict_example}, Type: {type(dict_example)}")


# SET TYPES
# ---------

# Set - unordered collection of unique items
set_example = {1, 2, 3, 4, 5}
print(f"Set: {set_example}, Type: {type(set_example)}")

# Frozen Set - immutable set
frozenset_example = frozenset([1, 2, 3, 4, 5])
print(f"Frozen Set: {frozenset_example}, Type: {type(frozenset_example)}")


# BOOLEAN TYPE
# -----------

# Boolean (bool) - True or False
bool_example_true = True
bool_example_false = False
print(f"Boolean True: {bool_example_true}, Type: {type(bool_example_true)}")
print(f"Boolean False: {bool_example_false}, Type: {type(bool_example_false)}")


# BINARY TYPES
# -----------

# Bytes - immutable sequence of bytes
bytes_example = b'hello'
print(f"Bytes: {bytes_example}, Type: {type(bytes_example)}")

# Bytearray - mutable sequence of bytes
bytearray_example = bytearray(b'hello')
print(f"Bytearray: {bytearray_example}, Type: {type(bytearray_example)}")

# Memoryview - memory view of an object
memoryview_example = memoryview(bytes_example)
print(f"Memoryview: {memoryview_example}, Type: {type(memoryview_example)}")
