# %% Imports
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import decomposition

# %% Exercise 1
iris_data = np.loadtxt('../01b/irisdata.txt', comments="%")
# x is a matrix with 50 rows and 4 columns
x = iris_data[0:50, 0:4]

n_feat = x.shape[1]
n_obs = x.shape[0]
print(f"Number of features: {n_feat} and number of observations: {n_obs}")

# %% Exercise 2
sep_l = x[:, 0]
sep_w = x[:, 1]
pet_l = x[:, 2]
pet_w = x[:, 3]

# Compute variance
# Use ddof = 1 to make an unbiased estimate
var_sep_l = sep_l.var(ddof=1)
var_sep_w = sep_w.var(ddof=1)
var_pet_l = pet_l.var(ddof=1)
var_pet_w = pet_w.var(ddof=1)


# %% Exercise 3
# TODO check result

def cov(a, b):
    a = np.array(a)
    b = np.array(b)
    n = len(a)
    sum_ab = np.sum(np.add(a, b))
    return (1 / n - 1) * sum_ab


print(cov(sep_l, pet_l))

print(cov(sep_l, sep_w))

# %% Exercise 4
plt.figure()  # Added this to make sure that the figure appear
# Transform the data into a Pandas dataframe
d = pd.DataFrame(x, columns=['Sepal length', 'Sepal width',
                             'Petal length', 'Petal width'])
sns.pairplot(d)
plt.show()

# %% Exercise 5
#TODO check result
mn = np.mean(x, axis=0)
data = x - mn

cov_mn = (1 / len(data) - 1) * np.matmul(data.T, data)

print(cov_mn)

c_x = np.cov(data)

# %% Exercise 6 - PCA

values, vectors = np.linalg.eig(c_x)

# %% Exercise 7
v_norm = values / values.sum() * 100
plt.plot(v_norm)
plt.xlabel('Principal component')
plt.ylabel('Percent explained variance')
plt.ylim([0, 100])
plt.show()

# %% Exercise 8
pc_proj = vectors.T.dot(data.T)

#%% Exercise 9 - PCA
pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_
data_transform = pca.transform(data)

print(data_transform)
