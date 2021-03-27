import numpy as np
import matplotlib.pyplot as plt

# %%

np.random.seed(12)

# define Ising model aprams
# system size
L = 40
N = 600

# create 10000 random Ising states
states = np.random.choice([-1, 1], size=(N, L))


def ising_energies(states):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    L = states.shape[1]
    J = np.zeros((L, L))
    for i in range(L):
        J[i, (i + 1) % L] = -1.0  # interaction between nearest-neighbors

    # compute energies
    E = np.einsum('...i,ij,...j->...', states, J, states)

    return E


# calculate Ising energies
energies = ising_energies(states)
energies
#%%
J = np.zeros((L, L))
J
# %%
# reshape Ising states into RL samples: S_iS_j --> X_p
states = np.einsum('...i,...j->...ij', states, states)
shape = states.shape
shape
#%%
states = states.reshape((shape[0], shape[1] * shape[2]))
# build final data set
Data = np.c_[states, energies.reshape(N,1)]
states
#%%
a=np.array([[[1,2,4,7,3],[2,1,4,2,6]]])
a.shape[2]

#%%
import pandas as pd
from sklearn.model_selection import train_test_split

df1 = pd.DataFrame(Data)
train_set, test_set = train_test_split(df1, test_size=0.2, random_state=42)
train_set
#%%
data_spins = train_set.drop([1600], axis=1, inplace=False)
data_label = train_set[1600].copy()
#%%
data_spins, data_label
#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, SGDRegressor

#%%
lin_reg = LinearRegression()
lin_reg.fit(data_spins, data_label)
#%%
lin_reg.get_params()
#%%
#prediction = lin_reg.predict([data_spins.iloc[20]])
#prediction
#data_label.iloc[20]
#%%
#lin_predict = lin_reg.predict(data_spins)
#lin_mse = mean_squared_error(lin_predict, data_label)
#lin_mse
#%%
ridge_reg = Ridge()  # alpha=1
#ridge_reg.get_params()
ridge_reg.fit(data_spins, data_label)
predict = ridge_reg.predict(data_spins)
ridg_mse = mean_squared_error(predict, data_label)
ridg_mse
#%%
# Seems we have overfitting, It could be solved, first, let us examine it with cross_val_score
#%%
ridge_score = cross_val_score(ridge_reg, data_spins, data_label,
                              scoring='neg_mean_squared_error',
                              cv=5)
print(-ridge_score)
ridge_rmse = np.sqrt(-ridge_score)
ridge_rmse
#%% It shows high mse, let's use hyperparameter tuning
#%%
param_grid = [{
    'alpha' : np.logspace(-4, 5, 10),
    #'copy_X': True, 'fit_intercept': True,
    #'max_iter': None, 'normalize': False, 'random_state': None
    }]

grid_search_ridge = GridSearchCV(ridge_reg, param_grid, cv=3,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search_ridge.fit(data_spins, data_label)
#%%
grid_search_ridge.best_params_
#%%
ridge_model = grid_search_ridge.best_estimator_
ridge_score = cross_val_score(ridge_model, data_spins, data_label,
                                   scoring='neg_mean_squared_error',
                                   cv=5)
ridge_rmse = np.sqrt(-ridge_score)
ridge_rmse
#%%
print(grid_search_ridge.best_params_)
#%%
ridge_model = grid_search_ridge.best_estimator_
j_ridge = ridge_model.coef_
j_ridge
#%%
#cmaps['Perceptually Uniform Sequential'] = [
#           'viridis', 'plasma', 'inferno', 'magma', 'cividis']
#
#cmaps['Sequential'] = [
#            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

plt.imshow(j_ridge.reshape(L, L), #cmap='YlOrRd'
           )
plt.title(r"Ridge  $\alpha=0.0001$")
plt.show()

#%%
ridge_reg.get_params()
#%%
lasso_reg = Lasso()  # alpha=1
param_grid = [{
    'alpha' : np.logspace(-4, 5, 10),
    #'copy_X': True, 'fit_intercept': True,
    #'max_iter': None, 'normalize': False, 'random_state': None
    }]

grid_search_lasso = GridSearchCV(lasso_reg, param_grid, cv=3,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search_lasso.fit(data_spins, data_label)
#%%
grid_search_lasso.best_params_
#%%
lasso_model = grid_search_lasso.best_estimator_
j_lasso = lasso_model.coef_
j_lasso
#%%
plt.imshow(j_lasso.reshape(L, L))
plt.title(r"LASSO  $\alpha=0.001$")
plt.show()
#%%
final_lasso_model = grid_search_lasso.best_estimator_
lasso_best_score = cross_val_score(final_lasso_model, data_spins, data_label,
                                   scoring='neg_mean_squared_error',
                                   cv=5)
lasso_best_rmse = np.sqrt(-lasso_best_score)
lasso_best_rmse
#%%