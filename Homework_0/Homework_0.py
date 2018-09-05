
# coding: utf-8

# 
# Author: Yinsen Miao

# ## Task 1

# In[1]:


get_ipython().system(u'conda info')


# ## Task 2

# In[2]:


import numpy as np
import scipy.linalg
# let us sample a array 5 by 5 a from a standard normal distribution
np.random.seed(2018)
a = np.random.normal(0, 1, (5,5))


# In[3]:


a.ndim


# In[4]:


a.size


# In[5]:


a.shape


# In[6]:


"nrows: %02d, ncols: %02d"%(a.shape[0], a.shape[1])


# In[7]:


# samole a 2 by 2 matrix b and construct a block matrix
b = np.random.normal(0, 1, (2, 2))
c = np.block([[b, b], [b, b]])  # construct a block matrix
print(c)


# In[8]:


a[-1] # access the last element of matrix a


# In[9]:


a[1, 4] # access element of the second row and the fifth column


# In[10]:


a[1]    # acess the entire second row


# In[11]:


a[0:5]  # access the first five rows of a


# In[12]:


a[-5:]  # access the last five rows of a


# In[13]:


# read-only access
a[0:3][:, 1:4]  # access rows from 1 to 3 and column from 2 to 4


# In[14]:


a[np.ix_([1, 3, 4], [0, 2])]


# In[15]:


# access ever other row of a starting from the third row 
# and going to the twenty-first
a[2:21:2, :]


# In[16]:


# access every other row of a starting from the first row
a[::2, :]


# In[17]:


# a with the rows in the reverse order
a[::-1, :]


# In[18]:


# append the firs row of array a to the last row of a
a[np.r_[:len(a), 0]]


# In[19]:


# transpose of a
a.transpose()


# In[20]:


# conjugate transpose of a
a.conj().T


# In[21]:


# matrix multiply
a.dot(a)


# In[22]:


# element-wise multiply
a * a


# In[23]:


# element-wise division
a / a


# In[24]:


# element-wise exponentiation
a ** 3


# In[25]:


# element-wise comparison
(a > 0.5)


# In[26]:


# find the index where (a > 0.5)
np.nonzero(a > 0.5)


# In[27]:


# extract the columns of a where vector v > 0.5
np.random.seed(2018)
v = np.random.normal(0, 1, (5))
v


# In[28]:


a[:, np.nonzero(v > 0.5)[0]]


# In[29]:


a[:, v.T > 0.5]


# In[30]:


# zero out element of a less than 0.5
a[a < 0.5] = 0
a


# In[31]:


# zero out element of a less than 0.5
a * (a > 0.5)


# In[32]:


# set a to the same scalar value 3
a[:] = 3
a


# In[33]:


# numpy assign by reference
np.random.seed(2018)
x = np.random.normal(0, 1, (2, 2))
y = x.copy()
y


# In[34]:


# numpy slices by reference
y = x[1,:].copy()
y


# In[35]:


# turn array to a vector and this operation forces a copy
y = x.flatten()
y


# In[36]:


# create an increasing vector
np.r_[1:11.]


# In[37]:


# create an increasing vector
np.arange(1., 11.)


# In[38]:


# create an increasing column vector
np.arange(1., 11.)[:, np.newaxis]


# In[39]:


# create two dimensional zero arrays
np.zeros((3, 4))


# In[40]:


# create three dimensional zero arrays
np.zeros((3, 4, 5))


# In[41]:


# 3 by 3 identity matrix
np.eye(3)


# In[42]:


# diagnal of matrix a
np.diag(a)


# In[43]:


# main diagnal of matrix a
np.diag(a, 0)


# In[44]:


# create a random matrix of 3 by 4
np.random.rand(3, 4)


# In[45]:


# equally spaced elements between 1 and 3 inclusive
np.linspace(1, 3, 4)


# In[46]:


# create two 2D arrays: one of x values and the other of y values
np.mgrid[0:9.0, 0:6.0]


# In[47]:


np.ogrid[0:9.0, 0:6.0]


# In[48]:


np.meshgrid([1,2,4], [2,4,5])


# In[49]:


# evaluate a function on a grid
np.ix_([1,2,4], [2,4,5])


# In[50]:


# create a 2 by 2 copies of x
np.tile(x, (2, 2)) 


# In[51]:


# concatenate the columns of x and x
np.concatenate((x, x), 1)


# In[52]:


# concatenate the rows of x and x
np.concatenate((x, x), 0)


# In[53]:


# find maximum element of array x
x.max()


# In[54]:


# maximum element of each column of matrix a
x.max(0)


# In[55]:


# maximum element of each row of matrix a
x.max(1)


# In[56]:


# compare a and b element-wise and return the maximum value per pair
np.random.seed(2018)
b = np.random.normal(3, 2, (5, 5))
np.maximum(a, b)


# In[57]:


# L2 norm of vector v
np.linalg.norm(v)


# In[58]:


# element-wise logic and
np.logical_and(a, b)


# In[59]:


# element-wise logic or
np.logical_or(a, b)


# In[60]:


# bitwise logic and
5 & 5


# In[61]:


# bitwise logic or
5 | 12


# In[62]:


# inverse of matrix
np.linalg.inv(x)


# In[63]:


# pseudo-inverse of matrix
np.linalg.pinv(x)


# In[64]:


# rank of a 2D matrix
np.linalg.matrix_rank(x)


# In[65]:


# least square solver 
np.random.seed(2018)
a = np.random.normal(0, 1, (3, 3))
b = np.random.normal(0, 1, (3, 1))
np.linalg.solve(a, b)


# In[66]:


# svd of a
U, S, Vh = np.linalg.svd(a)
U, S, Vh


# In[67]:


# upper Cholesky factor of matrix a
np.linalg.cholesky(a + 2*np.eye(3)).T


# In[68]:


# eigenvalues and eigenvectors of a
D, V = np.linalg.eig(a)
D, V


# In[69]:


# QR decomposition of matrix a
Q, R = scipy.linalg.qr(a)
Q, R


# In[70]:


# LU decomposition of matrix a
P, L, U = scipy.linalg.lu(a)
P, L, U


# In[71]:


# sort the matrix by row
np.sort(x)


# In[72]:


# multilinear regression
np.linalg.lstsq(a, b, rcond=None)


# In[73]:


# downsample with low-pass filtering
np.random.seed(2018)
x = np.random.normal(0, 1, (1000, 1))
#scipy.signal.resample(x, 20)


# In[74]:


# unique element in array a
np.unique(a)


# In[75]:


# squeeze
s = np.array([[1],[2],[3]])
s


# In[76]:


s.squeeze()


# ## Task 3

# In[77]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
# plot task 3
plt.plot([1,2,3,4], [1,2,7,14])
plt.axis([0, 6, 0, 20])
plt.show()


# ## Task 4

# In[78]:


# plot the histogram of a normal distribution
samples = np.random.normal(0, 1, 1000)
plt.hist(samples)
plt.show()


# ## Task 5
# 
# My github account is **yinsenm**.

# ## Task 6
# The github link to my reposistory is https://github.com/yinsenm/COMP576  .
