import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, cross_val_score

from random import shuffle


# From: https://www.kaggle.com/siddhrath/lightgbm-full-pipeline-model
def reduce_mem_usage(df, verbose=True):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage().sum() / 1024**2    
	for col in df.columns:
		col_type = df[col].dtypes
		if col_type in numerics:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)  
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)    
	end_mem = df.memory_usage().sum() / 1024**2
	if verbose: 
		print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
	return df


def get_portion(df, r=0.01):
	mols = list(set(df.molecule_name))
	shuffle(mols)
	return df[df.molecule_name.isin(mols[:int(len(mols) * r)])].copy()


	# From: https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
def map_atom_info(df, df_structures, atom_idx):
	df = pd.merge(df, df_structures, how = 'left',
				  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
				  right_on = ['molecule_name',  'atom_index'])
	
	df = df.drop('atom_index', axis=1)
	df = df.rename(columns={'atom': f'atom_{atom_idx}',
							'x': f'x_{atom_idx}',
							'y': f'y_{atom_idx}',
							'z': f'z_{atom_idx}'})
	return df


# From: https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
def get_dists(df, df_structures):
	df = map_atom_info(df, df_structures, 0)
	df = map_atom_info(df, df_structures, 1)
	train_p_0 = df[['x_0', 'y_0', 'z_0']].values
	train_p_1 = df[['x_1', 'y_1', 'z_1']].values
	df['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
	return df


def atoms_per_molecule(df):
	def core_atoms_per_molecule(df, atom):
		qry = df.query(f'atom_1 == "{atom}"')
		counts = qry.groupby('molecule_name')['atom_index_1'].nunique().to_dict()
		df['n_' + atom] = df['molecule_name'].apply(lambda x: counts.get(x, 0))
		return df
	for atom in df['atom_1'].unique():
		df = core_atoms_per_molecule(df, atom)
	return df


def plist(foolist):
    for a,b,c in zip(foolist[::3],foolist[1::3],foolist[2::3]):
        print('{:<20}{:<20}{:<}'.format(a,b,c))


def importances(feats, model):
    return pd.DataFrame(index=feats, data=model.feature_importances_,
                        columns=['importance']).sort_values('importance')[::-1]


def encode_cols(df, cols):
    for c in cols:
        encoded = pd.get_dummies(df[c], prefix=c, sparse=True)
        df = df.join(encoded)
    return df.drop(cols, axis=1)


# In[ ]:


df_orig = pd.read_csv('../input/train.csv', index_col='id')
df_structures = pd.read_csv('../input/structures.csv');
df_scc = pd.read_csv('../input/scalar_coupling_contributions.csv');


# In[ ]:


join_keys = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type']
df_orig = pd.merge(df_orig, df_scc, how = 'left', left_on= join_keys, right_on= join_keys)
del df_scc


# In[ ]:


df_orig = reduce_mem_usage(df_orig)
df_structures = reduce_mem_usage(df_structures)


# In[ ]:


df = df_orig #get_portion(df_orig, r=0.05)
gc.collect()


# # Feature Engineering

# In[ ]:


df = get_dists(df, df_structures)
df = atoms_per_molecule(df)


# In[ ]:


# Descriptive Stats
def get_descriptive_stats(df):
#     df['mean_dist_by_atom'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
#     df['max_dist_by_atom'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('max')
#     df['min_dist_by_atom'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
#     df['std_dist_by_atom'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std').fillna(0)
    
    df['mean_dist_by_mol'] = df.groupby(['molecule_name'])['dist'].transform('mean')
    df['max_dist_by_mol'] = df.groupby(['molecule_name'])['dist'].transform('max')
    df['min_dist_by_mol'] = df.groupby(['molecule_name'])['dist'].transform('mean')
    df['std_dist_by_mol'] = df.groupby(['molecule_name'])['dist'].transform('std').fillna(0)
    return df
df = get_descriptive_stats(df)


# In[ ]:


def get_num_neighbours(df):
    for atom in df.atom_1.unique():
        df_nearby = pd.DataFrame(df.query(f'atom_1 == "{atom}"').groupby(['molecule_name', 'atom_index_1', 'atom_0'])['atom_0'].size())
        df_nearby = df_nearby.fillna(0)
        df_nearby.columns = [f'nearby_{atom}']
        df = pd.merge(df, df_nearby, how='left', on=['molecule_name', 'atom_index_1', 'atom_0']).fillna(0)
    return df
df = get_num_neighbours(df)


# # Preparation for Training

# In[ ]:


all_feats = [
#     'molecule_name',
#     'atom_index_0',
#     'atom_index_1',
    'type',
#     'scalar_coupling_constant',
#     'fc',
#     'sd',
#     'pso',
#     'dso',
#     'atom_0',
    'x_0',
    'y_0',
    'z_0',
    'atom_1',
    'x_1',
    'y_1',
    'z_1',
    'dist',
    'n_C',
    'n_H',
    'n_N',
#     'mean_dist_by_atom',
#     'max_dist_by_atom',
#     'min_dist_by_atom',
#     'std_dist_by_atom',
    'mean_dist_by_mol',
    'max_dist_by_mol',
    'min_dist_by_mol',
    'std_dist_by_mol',
    'nearby_C',
    'nearby_N',
    'nearby_H'
]
categoricals = ['type', 'atom_1']
target = ['fc']


# # Encode Categoricals 

# In[ ]:


categoricals = ['type', 'atom_1']
feats = list(set(all_feats) - set(categoricals))


# # Predicting Constants

# In[ ]:


def constant_model(X, feats, constant):
    assert constant in df.columns
    print(f'Returning model to generate {constant}.')
    rf = RandomForestRegressor(n_jobs=-1)
    gss = GroupShuffleSplit(n_splits=3)
    scores = cross_val_score(rf, X[feats], y, scoring='neg_mean_absolute_error',
                            groups=X['molecule_name'], cv=gss, n_jobs=-1)
    rf.fit(X[feats], y)
    print(np.mean(np.log(-scores)))
    return rf
from joblib import dump, load
def save_model(model, fname):
    dump(model, f'{fname}.joblib') 


# In[ ]:


# Load Main Data and Test Data and Compute Constants now to Delete Model Right After (and save RAM)
def featurize(df_, categoricals=None):
    # Feature Engineering
    df_ = get_dists(df_, df_structures)
    df_ = atoms_per_molecule(df_)
    df_ = get_descriptive_stats(df_)
    df_ = get_num_neighbours(df_)
    # Encode Categoricals
    df_ = encode_cols(df_, categoricals)
    assert not df.isna().all().any(), "NaNs present in DataFrame"
    return df_


# In[ ]:


# Split for training meta-features
X = df.loc[:, feats + ['molecule_name']]


# In[ ]:


target = 'fc'
print('TARGET:', target[0])
print(f'FEATURES (n = {len(X.columns)}):')
plist(X.columns)
print(75*'-')
y = df.loc[:, [target]]
fc_model = constant_model(X, feats, target)
save_model(fc_model, 'fc_model')
# Generate Meta-Feature on Train Data
df[f"{target}_pred"] = fc_model.predict(X[feats])
# Generate Meta-Feature on Submission Data
X_final[f"{target}_pred"] = fc_model.predict(X_final[feats])
del fc_model
gc.collect()


# In[ ]:


target = 'sd'
print('TARGET:', target[0])
print(f'FEATURES (n = {len(X.columns)}):')
plist(X.columns)
print(75*'-')
y = df.loc[:, [target]]
sd_model = constant_model(X, feats, target)
save_model(sd_model, 'sd_model')
# Generate Meta-Feature on Train Data
df[f"{target}_pred"] = sd_model.predict(X[feats])
# Generate Meta-Feature on Submission Data
X_final[f"{target}_pred"] = sd_model.predict(X_final[feats])
del sd_model
gc.collect()


# In[ ]:


target = 'pso'
print('TARGET:', target[0])
print(f'FEATURES (n = {len(X.columns)}):')
plist(X.columns)
print(75*'-')
y = df.loc[:, [target]]
pso_model = constant_model(X, feats, target)
save_model(pso_model, 'pso_model')
# Generate Meta-Feature on Train Data
df[f"{target}_pred"] = pso_model.predict(X[feats])
# Generate Meta-Feature on Submission Data
X_final[f"{target}_pred"] = pso_model.predict(X_final[feats])
del pso_model
gc.collect()


# In[ ]:


target = 'dso'
print('TARGET:', target[0])
print(f'FEATURES (n = {len(X.columns)}):')
plist(X.columns)
print(75*'-')
y = df.loc[:, [target]]
dso_model = constant_model(X, feats, target)
save_model(dso_model, 'dso_model')
# Generate Meta-Feature on Train Data
df[f"{target}_pred"] = dso_model.predict(X[feats])
# Generate Meta-Feature on Submission Data
X_final[f"{target}_pred"] = dso_model.predict(X_final[feats])
del dso_model
gc.collect()


# In[ ]:


meta_feats = ['fc_pred', 'sd_pred', 'pso_pred', 'dso_pred']
meta_orig = ['fc', 'sd', 'pso', 'dso']


# In[ ]:


del X, y
gc.collect()


# ## Train Model (Scalar Coupling Constant) using meta-feats to see if overfitting

# In[ ]:


# Split Target
final_targ = ['scalar_coupling_constant']
# Reduce Memory
df = reduce_mem_usage(df)
X_ = df.loc[:, feats + meta_feats + ['molecule_name']]
y_ = df.loc[:, final_targ]
gc.collect()


# In[ ]:


model = RandomForestRegressor(n_jobs=-1)
gss = GroupShuffleSplit(n_splits=3)
train_idxs, test_idxs = next(gss.split(X_['molecule_name'], y_, groups=X_['molecule_name']))
X_train, X_test, y_train, y_test = X_.iloc[train_idxs], X_.iloc[test_idxs], y_.iloc[train_idxs], y_.iloc[test_idxs]


# In[ ]:


model.fit(X_train[feats + meta_feats], y_train)
y_pred = model.predict(X_test[feats + meta_feats])
np.log(mean_absolute_error(y_test, y_pred))


# In[ ]:


# Retrain on all data to save for later analysis
model.fit(X_[feats + meta_feats], y_)
y_pred = model.predict(X_[feats + meta_feats])


# # Final Model

# In[ ]:


# Now Train Final Model, and use original constants instead of meta_features
X_ = df.loc[:, feats + meta_orig]
y_ = df.loc[:, final_targ]
model.fit(X_[feats + meta_orig], y_)


# In[ ]:


y_pred = model.predict(df.loc[:, feats + meta_feats])
np.log(mean_absolute_error(y_, y_pred)) # See performance on meta features


# In[ ]:


del X_, y_, X_train, X_test, y_train, y_test, df_orig
gc.collect()


# # Submission

# In[ ]:


X_final_ids = X_final.loc[:, 'id']
X_final = X_final.drop(['id'], axis=1)

# Prepare for Predictions
X_final = X_final.loc[:, feats + meta_feats]
X_final = reduce_mem_usage(X_final)
gc.collect()


# In[ ]:


X_final = pd.DataFrame(np.nan_to_num(X_final[feats + meta_feats]), columns=feats + meta_feats)
print(X_final.columns)
pred = model.predict(X_final)


# In[ ]:


final_submission = pd.concat([X_final_ids, pd.Series(pred)], axis=1)
final_submission.columns = ['id', 'scalar_coupling_constant']


# In[ ]:


final_submission.to_csv('submission.csv', header=True, index=False)

