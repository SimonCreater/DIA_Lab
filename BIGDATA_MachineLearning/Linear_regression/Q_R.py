
import numpy as np 

arr = np.array([[5, 11, -15], [12, 34, -51], 
				[-24, -43, 92]], dtype=np.int32) 

q, r = np.linalg.qr(arr) 
print("Decomposition of matrix:") 
print( "q=\n", q, "\nr=\n", r)
