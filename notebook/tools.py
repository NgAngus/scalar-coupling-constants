import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import gc

from scipy.stats import kurtosis
from random import shuffle
from joblib import dump, load
from itertools import compress

# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel


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
		print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
		 100 * (start_mem - end_mem) / start_mem))
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


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return np.nan_to_num(vector.div(np.sqrt(np.square(vector).sum(axis=1)), axis='rows'))


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return pd.Series(np.arccos(np.clip(np.einsum('ij,ij->i', v1_u, v2_u), -1.0, 1.0)))


def get_dipole_feats(df):
	if set(['dip_x', 'dip_y', 'dip_z']).issubset(set(df.columns)):
		print("Dipole information found...computing dipole features.")
		df['dip_mag'] = np.sqrt(np.square(df[['dip_x', 'dip_y', 'dip_z']]).sum(axis=1))
		df['has_dip'] = (df['dip_mag'] != 0).astype(int)
		df['dip_angle'] = angle_between(df[['dip_x', 'dip_y', 'dip_z']], df[['x_0', 'y_0', 'z_0']])
	return df.fillna(0)


def atoms_per_molecule(df):
	def core_atoms_per_molecule(df, atom):
		qry = df.query(f'atom_1 == "{atom}"')
		counts = qry.groupby('molecule_name')['atom_index_1'].nunique().to_dict()
		df['n_' + atom] = df['molecule_name'].apply(lambda x: counts.get(x, 0))
		return df
	for atom in df.atom_1.unique():
		df = core_atoms_per_molecule(df, atom)
	return df


def plist(foolist):
	for a,b,c in zip(foolist[::3],foolist[1::3],foolist[2::3]):
		print('{:<20}{:<20}{:<}'.format(a,b,c))


def importances(feats, model):
	return pd.DataFrame(index=feats, data=model.feature_importances_,
						columns=['importance']).sort_values('importance')[::-1]


def encode_cols(df, cols):
	if not cols:
		print('No columns given to encode.')
		return df
	for c in cols:
		encoded = pd.get_dummies(df[c], prefix=c, sparse=True)
		df = df.join(encoded)
	return df.drop(cols, axis=1)


def get_nth_mindist(df, n=2, atom_index=0):
    grouping = ['molecule_name', f'atom_index_{atom_index}']
    df_mindist2 = df.groupby(grouping)['dist'].nsmallest(n).groupby(grouping).last().to_frame().reset_index()
    df_mindist2.columns = grouping + [f'atom_{atom_index}_min_dist{n}']
    return pd.merge(df, df_mindist2, how='left', on=grouping).fillna(0)


# Descriptive Stats
def get_descriptive_stats(df):
    aggdict = {
        'mol_min_dist': min,
        'mol_max_dist': max,
        'mol_mean_dist': np.mean,
        'mol_med_dist': np.median,
        'mol_kur_dist': kurtosis,
        'mol_std_dist': np.std,
    }
    aggdict0 = {
        'atom_0_min_dist': min,
        'atom_0_max_dist': max,
        'atom_0_mean_dist': np.mean,
        'atom_0_med_dist': np.median,
        'atom_0_kur_dist': kurtosis,
        'atom_0_std_dist': np.std,
    }
    aggdict1 = {
        'atom_1_min_dist': min,
        'atom_1_max_dist': max,
        'atom_1_mean_dist': np.mean,
        'atom_1_med_dist': np.median,
        'atom_1_kur_dist': kurtosis,
        'atom_1_std_dist': np.std,
    }
    desc_stats = df.groupby(['molecule_name'])['dist'].agg(aggdict).reset_index().fillna(0)
    desc_stats0 = df.groupby(['molecule_name', 'atom_index_0'])['dist'].agg(aggdict0).reset_index().fillna(0)
    desc_stats1 = df.groupby(['molecule_name', 'atom_index_1'])['dist'].agg(aggdict1).reset_index().fillna(0)

    
    df = pd.merge(left=df, right=desc_stats, how='left', on=['molecule_name'])
    df = pd.merge(left=df, right=desc_stats0, how='left', on=['molecule_name', 'atom_index_0'])
    df = pd.merge(left=df, right=desc_stats1, how='left', on=['molecule_name', 'atom_index_1'])
    return df


def get_num_neighbours(df):
	for atom in df.atom_1.unique():
		df_nearby = pd.DataFrame(df.query(f'atom_1 == "{atom}"').groupby(['molecule_name', 
			'atom_index_1', 'atom_0'])['atom_0'].size())
		df_nearby = df_nearby.fillna(0)
		df_nearby.columns = [f'nearby_{atom}']
		df = pd.merge(df, df_nearby, how='left', on=['molecule_name', 'atom_index_1', 'atom_0'])
	return df.fillna(0)


def featurize(df_, df_structures):
	# Feature Engineering
	df_ = get_dists(df_, df_structures)
	df_ = get_nth_mindist(df_, n=2)
	df_ = get_nth_mindist(df_, n=3)
	df_ = get_dipole_feats(df_)
	df_ = atoms_per_molecule(df_)
	df_ = get_descriptive_stats(df_)
	df_ = get_num_neighbours(df_)
	assert not df_.isna().all().any(), "NaNs present in DataFrame"
	return df_


def save_model(model, fname):
	dump(model, f'{fname}.joblib')


def constant_model(X, y, feats, constant, verbose=True, cv=True):
    if verbose:
        print('TARGET:', constant)
    print(f'FEATURES (n = {len(feats)}):')
    plist(feats)
    print(75*'-')
    print(f'Returning model to generate {constant}.')
    rf = RandomForestRegressor(n_jobs=-1)
    if cv:
        gss = GroupKFold(n_splits=3)
        scores = cross_val_score(rf, X[feats], y, scoring='neg_mean_absolute_error', groups=X['molecule_name'], cv=gss, n_jobs=-1)
        print("CV-score", np.mean(np.log(-scores)))
    rf.fit(X[feats], y)
    return rf


def select_features(model, X, y, feats, threshold=0.0001):
    sfm = SelectFromModel(model, threshold=threshold)
    sfm.fit(X[feats], y)
    n_features = sfm.transform(X[feats]).shape[1]
    while 2*n_features > len(feats):
        sfm.threshold += 0.0001
        X_transform = sfm.transform(X[feats])
        n_features = X_transform.shape[1]
        
    print(f'Reduced from {len(feats)} to {n_features}.')
    feats_selected = list(compress(feats, sfm.get_support()))
    print(feats_selected)
    return feats_selected

    