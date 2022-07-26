
import numpy as np
import allensdk.brain_observatory.stimulus_info as stim_info

from sklearn.decomposition import PCA

# global variables  
n_divs = 30
n_frames = 900
n_repeats = 10
n_sessions = 3
frames_per_repeat = int(n_frames/n_divs)


def get_datasets(boc, container_id):

    exps_container = boc.get_ophys_experiments(experiment_container_ids= [container_id], stimuli=[stim_info.NATURAL_MOVIE_ONE])
    ids = [exp['id'] for exp in exps_container]

    age_days = []
    for id in ids:
        dataset = boc.get_ophys_experiment_data(id)
        age_days.append(dataset.get_metadata()['age_days'])
    days_order = np.argsort(age_days)

    # create datasets in one container:
    datasets = []
    ordered_days = []
    for i in range(len(ids)):
        dataset = boc.get_ophys_experiment_data(ids[days_order[i]])
        ordered_days.append(dataset.get_metadata()['age_days'])
        datasets.append(dataset)

    ordered_days = np.array(ordered_days) - ordered_days[0]

    return datasets, ordered_days




def get_response_vals(data_set):
    '''
    input     : dataset
    output    : cell_ids, trace vectors, running speed
    '''

    _, dff_traces = data_set.get_dff_traces()
    cell_ids = data_set.get_cell_specimen_ids()

    stim_table = data_set.get_stimulus_table('natural_movie_one')
    running_speed = data_set.get_running_speed()[0]
    # pupil_size = data_set.get_pupil_size()[1] 

    n_cells = dff_traces.shape[0]

    dff_vals = np.zeros((n_repeats, n_divs, n_cells))
    # ppl_vals = np.zeros((n_repeats, n_divs))
    run_vals = np.zeros((n_repeats, n_divs))

    # for each repeat:
    for repeat_idx in range(n_repeats):
        # grab all frames belongs to the repeat
        repeat_frames = np.array(stim_table.query('repeat == @repeat_idx')['start'])
        # for each block:
        for div_idx in range(n_divs):
            # grab block frames
            div_repeat_idxs = repeat_frames[
                div_idx*frames_per_repeat:(div_idx+1)*frames_per_repeat]     
            # average over block frames:
            dff_vals[repeat_idx, div_idx] = np.mean(dff_traces[:, div_repeat_idxs], axis=1)
            # ppl_vals[repeat_idx, div_idx] = np.nanmean(pupil_size[div_repeat_idxs]) # there are NaNs
            run_vals[repeat_idx, div_idx] = np.mean(running_speed[div_repeat_idxs])
    
    return cell_ids, dff_vals, run_vals # ppl_vals


def shared_idx(in_session_cell, shared_cell):
    
    sorter = np.argsort(in_session_cell)
    in_session_idx = sorter[
        np.searchsorted(in_session_cell, shared_cell, sorter = sorter)]

    return in_session_idx



def align_response_vals(datasets):

    cell_ids_0, dff_vals_0, run_vals_0 = get_response_vals(datasets[0])
    cell_ids_1, dff_vals_1, run_vals_1 = get_response_vals(datasets[1])
    cell_ids_2, dff_vals_2, run_vals_2 = get_response_vals(datasets[2])

    shared_ids_01 = np.intersect1d(cell_ids_0, cell_ids_1)
    shared_ids = np.intersect1d(shared_ids_01, cell_ids_2)

    ids_idx_0 = shared_idx(cell_ids_0, shared_ids)
    ids_idx_1 = shared_idx(cell_ids_1, shared_ids)
    ids_idx_2 = shared_idx(cell_ids_2, shared_ids)

    dff_vals_0 = dff_vals_0[:, :, ids_idx_0]
    dff_vals_1 = dff_vals_1[:, :, ids_idx_1]
    dff_vals_2 = dff_vals_2[:, :, ids_idx_2]
        
    dff_vals_container = [dff_vals_0, dff_vals_1, dff_vals_2]
    run_vals_container = [run_vals_0, run_vals_1, run_vals_2]

    return dff_vals_container, run_vals_container



def calc_similarity(dff_vals):
    '''
    input: response vectors 
    output: correlation coef. matrix & angle matrix
    '''
    within_session_corrs = np.zeros((n_repeats*n_divs, n_repeats*n_divs))
    within_session_angles = np.zeros((n_repeats*n_divs, n_repeats*n_divs))

    for repeat_idx1 in range(n_repeats):
        for repeat_idx2 in range(n_repeats):
            # corrcoef for all blocks in repeat_1 and repeat_2:
            within_session_corrs[ 
                repeat_idx1*n_divs : (repeat_idx1+1)*n_divs, 
                repeat_idx2*n_divs : (repeat_idx2+1)*n_divs] = np.corrcoef(
                    # return 30x30 R-values:
                    dff_vals[repeat_idx1], dff_vals[repeat_idx2])[n_divs:, :n_divs] # for some reason corrcoef returns 4 copies of this matrix
            
            for div_idx1 in range(n_divs):
                for div_idx2 in range(n_divs):
                    # calculate angle 
                    # for resp_vec of block_1 in repeat_1 and
                    #  resp_vec in block_2 in repeat_2:
                    within_session_angles[
                        repeat_idx1 * n_divs + div_idx1, 
                        repeat_idx2 * n_divs + div_idx2] = get_align_angle(
                            dff_vals[repeat_idx1, div_idx1], 
                            dff_vals[repeat_idx2, div_idx2])

    return within_session_corrs, within_session_angles


def calc_mean_vecs(dff_vals_container):

    n_shared = dff_vals_container[0].shape[-1]
    mean_vecs = np.zeros((n_sessions, n_divs, n_shared))

    for session_idx in range(n_sessions):
        dff_vals_session = dff_vals_container[session_idx]
        mean_vecs[session_idx] = np.mean(dff_vals_session, axis=0)
    
    return mean_vecs


def effective_dimention(vars_explained):

    D = vars_explained.sum()**2/ (vars_explained**2).sum()
    
    return D


def PCA_for_stimuli(dff_vals_stimuli, n_pcas):

    pca_div = PCA(n_pcas)
    # fit on the 10 vecs:
    pca_div.fit(dff_vals_stimuli)
    
    # Removes ambiguity in PCA direction by making sure all means (along PC directions) are positive
    pca_dir_signs = np.sign(np.mean(np.matmul(dff_vals_stimuli, pca_div.components_.T), axis=0))
    for pca_idx, pca_dir in enumerate(pca_div.components_):
        pca_div.components_[pca_idx, :] = pca_dir_signs[pca_idx] * pca_dir        
    
    return pca_div



def do_PCA(dff_vals_container):

    n_shared = dff_vals_container[0].shape[-1]
    n_pcas = np.min((n_repeats, n_shared))
    # there are 3 list within this list, each list for all the blocks.
    pcas_session = [[] for _ in range(n_sessions)] 
    vars_session = np.zeros((n_sessions, n_divs, n_repeats))
    dims_session = np.zeros((n_sessions, n_divs))
    for session_idx in range(n_sessions):
        dff_vals_session = dff_vals_container[session_idx]
        # for each block:
        for div_idx in range(n_divs):
            pca_div =  PCA_for_stimuli(dff_vals_session[:, div_idx], n_pcas)
            vars_session[session_idx, div_idx, :] = pca_div.explained_variance_ratio_
            dim_var_space = effective_dimention(pca_div.explained_variance_ratio_)

            
            pcas_session[session_idx].append(pca_div)
            dims_session[session_idx][div_idx] = dim_var_space
    
    return pcas_session, vars_session, dims_session
        


# helper func:
def get_align_angle(x, y):
    'input: two vectors of same length; output: angle between vec in degrees'
    dot = np.dot(x,y)/(
         np.linalg.norm(x) * np.linalg.norm(y))
    if dot > 1.0:
         dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    
    return 180/np.pi * np.arccos(dot)


def drift_geometry(dff_vals_container, pcas_session, vars_session, mean_vecs):


    drift_pair_idxs = [[0, 1], [1, 2], [0, 2]]
    n_drifts = len(drift_pair_idxs)
    n_shared = dff_vals_container[0].shape[-1]

    # 1-2, 2-3, 1-3: 30 different stimuli, how the drift goes:
    drifts_session = np.zeros((n_drifts, n_divs, n_shared))
    n_pcas = np.min((n_repeats, n_shared))

    # magnitude of drift of PCs for the 30 stimuli:
    pca_drift_magnitude = np.zeros((n_drifts, n_divs, n_pcas))
    pca_drift_align = np.zeros((n_drifts, n_divs, n_pcas))
    pca1_var_exp2_session = np.zeros((n_drifts, n_divs, n_repeats))

    for drift_idx in range(n_drifts):
        session_idx1 = drift_pair_idxs[drift_idx][0]
        session_idx2 = drift_pair_idxs[drift_idx][1]

        drifts_session[drift_idx] = (mean_vecs[session_idx2] - 
                        mean_vecs[session_idx1])

        for div_idx in range(n_divs):
            # Drift in the PCA coordinates (of starting session)
            # proj of the drift, on the components of the first session
            pca_drift = np.matmul( 
                pcas_session[session_idx1][div_idx].components_,
                drifts_session[drift_idx, div_idx])

            for pca_idx in range(n_pcas):
                pca_drift_magnitude[drift_idx, div_idx, pca_idx] = np.abs(pca_drift[pca_idx])/ np.linalg.norm(
                            drifts_session[drift_idx, div_idx])
                
                pca_drift_align[drift_idx, div_idx, pca_idx] = get_align_angle(
                            drifts_session[drift_idx, div_idx],
                            pcas_session[session_idx1][div_idx].components_[pca_idx])
                        

                data1 = dff_vals_container[session_idx1][:, div_idx, :]
                data2 = dff_vals_container[session_idx2][:, div_idx, :]

                # Flow of variation
                pca1_data1 = np.matmul(data1, pcas_session[session_idx1][div_idx].components_.T)
                pca1_data2 = np.matmul(data2, pcas_session[session_idx1][div_idx].components_.T)
                
                pca1_data1_cov = np.cov(pca1_data1.T)
                # pca1_data1_vars = np.diag(pca1_data1_cov)
                # pca1_data1_vars_ratio = pca1_data1_vars/np.sum(pca1_data1_vars)
                pca1_data2_cov = np.cov(pca1_data2.T)
                pca1_data2_vars = np.diag(pca1_data2_cov)
                pca1_var_exp2_session[drift_idx, div_idx] = pca1_data2_vars/np.sum(pca1_data2_vars)

    pca1_var= np.zeros((n_sessions, n_divs, n_repeats))

    for session_idx in range(n_sessions):
        pca1_var[session_idx, :, :]  = vars_session[drift_pair_idxs[session_idx][0], :, :]

    return (pca1_var, pca_drift_magnitude, pca_drift_align, pca1_var_exp2_session)
