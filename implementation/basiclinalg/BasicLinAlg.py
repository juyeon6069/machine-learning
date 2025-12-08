import numpy as np

# look at accompanying practice_linear_alg.pdf to know what problems to code up

def linalg_operations(A,b,c,x,which_problem):
    AT = np.transpose(A)
    xT = np.transpose(x)
    
    colA = A.shape[0]
    rowA = A.shape[1] if A.ndim != 1 else 1
    
    colAT = rowA
    rowAT = colA
        
    colb = b.shape[0]   
    rowb = b.shape[1] if b.ndim != 1 else 1
    
    colx = x.shape[0]    
    rowx = x.shape[1] if x.ndim != 1 else 1
    
    colxT = rowx
    rowxT = colx
    
    if which_problem == '2a':
        if rowA != colx:
            return 'not possible'
        Ax = np.dot(A, x)
        return Ax
   
    if which_problem == '2b':
        if rowA != colb:
            return 'not possible'
        Ab = np.dot(A, b)
        return Ab
    
    if which_problem == '2c':
        if rowAT != colb:
            return 'not possible'
        ATb = np.dot(AT, b)
        return ATb
    
    if which_problem == '2d':
        if rowxT != colA:
            return 'not possible'
        xTA = np.dot(xT, A)
        return xTA
        
    if which_problem == '2e':
        if rowxT != colAT:
            return 'not possible'
        xTAT = np.dot(xT, AT)
        return xTAT

    if which_problem == '2f':
        norm_x = 0
        for i in range(0, len(x)):
            norm_x = norm_x + x[i] **2
        sqr_norm_x = np.sqrt(norm_x)
        return sqr_norm_x
    
    if which_problem == '2g':
        norm_b = 0
        for i in range(0, len(b)):
            norm_b = norm_b + b[i] **2
        sqr_norm_b = np.sqrt(norm_b)
        return sqr_norm_b ** 3
    
    if which_problem == '2h':
        if rowA != colx:
            return 'not possible'
        norm_u = 0
        Ax = np.dot(A, x)
        if Ax.shape != b.shape:
            return 'not possible'
        u = Ax - b
        colu = u.shape[0]
        for i in range(0, colu):
            norm_u = norm_u + u[i] **2
        sqr_norm_u = np.sqrt(norm_u)
        return sqr_norm_u ** 2
    
    
def linalg_gradients(A,b,c,x,which_problem):
    if which_problem == '4a':
        norm_x = np.poly1d([1, 0 , 0])
        deriv_x = norm_x.deriv()
        return deriv_x(x)
    
    if which_problem == '4b':
        AT = np.transpose(A)
        Ax = np.dot(A, x)
        deriv_x = 2 * np.dot(AT, Ax)
        return deriv_x
    
    if which_problem == '4c':
        cT = np.transpose(c)
        cTx = np.dot(cT, x)
        sqr_cTx = np.dot(cTx, cTx)
        deriv_x = 3 * np.dot(sqr_cTx, c)
        return deriv_x
    
    if which_problem == '4d':
        bT = np.transpose(b)
        Ax = np.dot(A, x)
        AT = np.transpose(A)
        ATb = np.dot(AT, b)
        bTAx = np.dot(bT, Ax)
        sqr_bTAx = np.dot(bTAx, bTAx)
        deriv_x = 3 * np.dot(sqr_bTAx, ATb)
        return deriv_x
        
    if which_problem == '4e':
        deriv_x = np.zeros_like(x)
        ux = b - np.dot(A, x)
        colA = A.shape[0]
        for i in range(colA):
            if ux[i] > 0:
                deriv_x[i] -= np.transpose(A[i, :])
        return deriv_x

    if which_problem == '4f':
        ux = x - 5
        norm_ux = 0
        for i in range(0, len(ux)):
            norm_ux = norm_ux + ux[i] ** 2
        sqr_norm_ux = np.sqrt(norm_ux)
        exp = np.exp(-1 * sqr_norm_ux ** 2)
        deriv_x = -2 * np.dot(ux, exp)
        return deriv_x
    
    if which_problem == '4g':
        bT = np.transpose(b)
        Ax = np.dot(A, x)
        AT = np.transpose(A)
        ATb = np.dot(AT, b)
        bTAx = np.dot(bT, Ax)
        exp = np.exp(-1 * bTAx)
        deriv_x = np.dot(exp, ATb) / (1+ exp)
        return deriv_x


 
