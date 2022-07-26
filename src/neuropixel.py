#!/usr/bin/env python
"""
created 07/21/22

@author Sharif Saleki

Implements loading and analyzing Allen Institute's Neuropixel dataset as a template for
representational drift analysis.
"""
import shutil
from pathlib import Path
from typing import Union
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


class Neuropixel:
    """
    Base class to read and analyze Allen's neuropixel dataset.

    Parameters
    ----------
    data_dir : str or Path
        Data directory of the experiment. If not specified, it will be {experiment_root}/data

    """
    def __init__(self, data_dir: Union[str, Path] = None):

        # Specify data directory
        if data_dir is None:
            self.DATA_DIR = Path().cwd().parent / "data" / "Ecephy"
        elif isinstance(data_dir, str):
            self.DATA_DIR = Path(data_dir)
        else:
            self.DATA_DIR = data_dir

        # Make a manifest path
        self.manifest_path = self.DATA_DIR / "manifest.json"

        # Load the project-level cache
        self._cache = EcephysProjectCache.from_warehouse(manifest=str(self.manifest_path))

        # Initiate data vars
        self._data = None
        self._session = None
        self._all_sessions = [
            int(f.name.split('_')[1]) for f in self.DATA_DIR.iterdir() if ('session' in f.name and 'csv' not in f.name)
        ]

        # Other parameters
        self.snr_thr_percent = 25

    @property
    def cache(self):
        """ Getter for experiment cache"""
        return self._cache

    @property
    def session(self):
        """ Getter for current session"""
        return self._session

    @session.setter
    def session(self, session_id):
        """ Selects a particular session as the current session of the data"""
        self._session = session_id

    def get_spikes_table(self):
        """
        Collects all the relevant parts of a data and puts them in a pandas DataFrame
        """
        pass

    def download_ecephy_data(self):
        """
        Downloads all the Neuropixel data.

        Returns
        -------
        None
        """
        # Session metadata
        sessions = self.cache.get_session_table()

        # load individual sessions in the table
        s = 1
        for sess, row in tqdm(sessions.iterrows()):

            # Print out what's happening
            print(f"Downloading session {sess} (#{s})")

            # to check for bad files
            truncated_file = True
            sess_path = self.DATA_DIR / f'session_{str(sess)}'

            # load the data
            while truncated_file:
                session = self.cache.get_session_data(sess)
                try:
                    print(session.specimen_name)
                    s += 1
                    truncated_file = False
                except OSError:
                    shutil.rmtree(sess_path)
                    print(" Truncated spikes file, re-downloading")

    def load_sessions(self, session_id: Union[int, list, str]):
        """
        Loads Neuropixel session data. The session_id can be specified as:

        1. An integer: session number
        2. A list: a bunch of session numbers
        3. A string: name of a session type. There are 2 types of sessions (brain_observatory1.1 and
        functional_connectivity). Can also specify "all" to get all the data.

        Parameters
        ----------
        session_id : int or list or str
            Session identifier

        Returns
        -------
        None
        """
        # Session metadata
        sessions = self.cache.get_session_table()

        # Load for a single session
        if isinstance(session_id, int):
            self._data[session_id] = self.cache.get_session_data(session_id)

        # Load for a list of sessions
        elif isinstance(session_id, list):
            for sess in session_id:
                self._data[sess] = self.cache.get_session_data(sess)

        # Load for a session type
        elif session_id in ["brain_observatory_1.1", "functional_connectivity"]:

            # select session type
            _sessions = sessions[sessions.session_type == session_id]

            # load individual sessions in the table
            for sess, row in _sessions.iterrows():
                self._data[sess] = self.cache.get_session_data(sess)

    def _get_roi_idx(self, roi: str):
        """
        Identifies units within a brain region

        Parameters
        ----------
        roi : str

        Returns
        -------
        np.array
        """
        # select unit snr
        q25 = np.percentile(self._data.units.snr, self.snr_thr_percent)

        roi_idx = self._data.units[
            (self._data.units.snr > q25) & (self._data.units.ecephys_structure_acronym == roi)
            ].index.values

        return roi_idx

    def _get_stim_idx(self, stim_type: str, condition: str):
        """

        Parameters
        ----------
        stim_type
        condition

        Returns
        -------

        """
        # Get the condition
        stim_id = self._data.stimulus_conditions.loc[
            self._data.stimulus_conditions.stimulus_name == stim_type
            ].index.values[0]

        # Get the indices
        stim_idx = self._data.stimulus_presentations.loc[
            self._data.stimulus_presentations.stimulus_condition_id == stim_id
            ].index.values

        return stim_idx

    def _pupil_run_average(self, bin_size: int):
        """
        Finds the average running speed and pupil size of all subjects in some time bin duration.

        Parameters
        ----------
        bin_size : int
            in seconds.

        Returns
        -------

        """
        # Make time bins
        time_bins = np.arange(0, 10000, bin_size)

        # Initiate stuff
        pupil_means = np.zeros((len(self._all_sessions), 10000))
        run_means = np.zeros((len(self._all_sessions), 10000))

        for s, session in enumerate(self._all_sessions):

            # load session metadata
            _data = self.cache.get_session_data(session)

            # get pupil and running data
            pupil = _data.get_pupil_data()
            run = _data.running_speed()
            run["speed"] = abs(run["velocity"])  # absolute value of speed

            # Get the pupil mean
            pupil_mean = np.array(
                [pupil.loc[(pupil.index > b) & (pupil.index < b + 1), "raw_pupil_area"].mean() for b in time_bins]
            )
            pupil_normal = stats.zscore(pupil_mean, nan_policy='omit')  # normalize
            pupil_normal[pupil_normal > 5] = 0  # get rid of very large values

            # get running speed mean for each bin
            run_mean = np.array(
                [run.loc[(run.start_time > b) & (run.end_time < b + 1), "speed"].mean() for b in time_bins]
            )
            # run_mean = run_mean[~np.isnan(run_mean)]
            run_normal = stats.zscore(run_mean, nan_policy='omit')  # normalize

            # save
            pupil_means[s] = pupil_mean
            run_means[s] = run_mean

        return pupil_means, run_means




