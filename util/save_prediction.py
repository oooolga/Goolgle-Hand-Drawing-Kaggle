import pandas as pd
import numpy as np
import pdb

def get_prediction_dataframe(pred_matrix, unique_labels):

	df =  pd.DataFrame(data=pred_matrix,
					   columns=['Id', 'Category'])

	for i in range(len(unique_labels)):
		df.loc[df.Category==i,'Category'] = unique_labels[i]

	return df

