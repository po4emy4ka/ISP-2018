import numpy as np
from lab2 import norma_m, norma_v

def algebraic_complements(a, i, j):
    a = np.delete(a, i, axis=0)
    a = np.delete(a, j, axis=1)
    #print(*a, sep="\n")
    #print()
    return (-1)**(i + j) * np.linalg.det(a)
    
def create_alpha(a):
    for i in range(len(a)):
        curr_ii = a[i][i] 
        for j in range(len(a)):
            if i == j:
                a[i][j] = 0
            else:
                a[i][j] = (-1) * a[i][j] / curr_ii
    return a

def create_betta(a, b):
    betta = np.zeros(len(a))
    for i in range(len(a)):
        betta[i] = b[i] / a[i][i]
    return betta


def reversing(a):
    # союзная
    al_compl = np.zeros(a.size).reshape(a.shape)
    
    for i in range(4):
        for j in range(4):
            al_compl[i][j] = algebraic_complements(a, i, j)
    
    # обратная
    reverse_a = al_compl.T / np.linalg.det(a)
    print("\n", "Algebraic Supplements:", "\n")
    print(*al_compl, sep="\n")
    print("\n", "Inverse matrix:", "\n")
    #print(norma_m(a))
    print(*reverse_a, sep="\n")
    return reverse_a
    
def gausssian(A, b):
    n =  len(A)
    for a_row in range(n - 1):
        for row in range(a_row + 1, n):
            multiplier = A[row][a_row] / A[a_row][a_row]
            #print(multiplier, row, a_row)
            for col in range(n):
                A[row][col] = A[row][col] - multiplier*A[a_row][col]
            
            b[row] = b[row] - multiplier*b[a_row]
        print()
    #print(*A, sep="\n")
    #print(*b, sep="\n")
    x = np.zeros(n)
    k = n - 1
    x[k]  = b[k] / A[k][k]
    
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k, k + 1:], x[k + 1:])) / A[k][k]
        k -= 1
    return x


def main():
    a = np.array([[2.9998, 0.209, 0.315, 0.281],
                 [0.163, 3.237, 0.226, 0.307],
                 [0.416, 0.175, 3.239, 0.159],
                 [0.287, 0.196, 0.325, 4.062]])
    b =  np.array([0.108, 0.426, 0.31, 0.084])
    
    
    a1 = np.array(a)
    b1 = np.array(b)
    revr_a = np.array(reversing(a))
    print()
    #print(revr_a)
    a = gausssian(a, b)
    for i in range(1, 5):
        print("x{} = ".format(i), round(a[i - 1], 4))
    norm_a = norma_m(a1)
    norm_b = norma_v(b1)
    norm_x = norma_v(a)
    norm_revr_a = norma_m(revr_a)
    print("Norma of matrix a: ", round(norm_a, 6))
    print("Norma of matrix b: ", round(norm_b, 6))
    print("Norma of matrix x: ", round(norm_x, 6))
    print("Norma of matrix a^(-1): ", round(norm_revr_a, 6))
    # delta - absolutnaya sigma - otnositelnaya    
    delta_b = 0.001
    sigma_b = delta_b / norm_b
    delta_x = norm_revr_a * delta_b
    sigma_x = delta_x / norm_x
    print("Relative error x of matrix : ", round(sigma_x, 6))
    print("Absolute error x of matrix : ", round(delta_x, 6))
if __name__ == "__main__":
    main()
