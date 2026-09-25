import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
addition = A + B
subtraction = A - B
multiplication = A * B         
matrix_mult = np.dot(A, B)  

print("\nMatrix Addition (A + B):")
print(addition)

print("\nMatrix Subtraction (A - B):")
print(subtraction)

print("\nElement-wise Multiplication (A * B):")
print(multiplication)

print("\nMatrix Multiplication (A . B):")
print(matrix_mult)

try:
    inverse_A = np.linalg.inv(A)
    print("\nInverse of Matrix A:")
    print(inverse_A)
except np.linalg.LinAlgError:
    print("\nMatrix A is singular, inverse does not exist.")

try:
    inverse_B = np.linalg.inv(B)
    print("\nInverse of Matrix B:")
    print(inverse_B)
except np.linalg.LinAlgError:
    print("\nMatrix B is singular, inverse does not exist.")


transpose_A = np.transpose(A)
transpose_B = np.transpose(B)
                                                       
print("\nTranspose of Matrix A:")
print(transpose_A)

print("\nTranspose of Matrix B:")
print(transpose_B)  
