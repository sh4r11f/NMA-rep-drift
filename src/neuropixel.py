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
        """ Selects a particular session as the current session of the experiment and loads its data"""
        self._session = session_id
        self._data = self.cache.get_session_data(session_id)

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

    def _get_stim_idx(self, stim_type: str):
        """
        Retrieves indices of repeated stimulus presentation.

        Parameters
        ----------
        stim_type : str
            Any stimulus that's not a movie!

        Returns
        -------
        np.array : (n_repeats,)
        """
        # Select the condition
        stim_id = self._data.stimulus_conditions.loc[
            self._data.stimulus_conditions.stimulus_name == stim_type
            ].index.values[0]

        # Get the indices
        repeat_idx = self._data.stimulus_presentations.loc[
            self._data.stimulus_presentations.stimulus_condition_id == stim_id
            ].index.values

        return repeat_idx

    def _get_movie_idx(self, movie_type: str, sec: int):
        """
        Retrieves index of presentations for the first frame of a movie in a second.

        Parameters
        ----------
        movie_type : str
            Which movie

        sec : int
            Which second.

        Returns
        -------
        np.array : (n_repeats,)
        """
        # Find the max number of frames
        max_frame = self._data.stimulus_conditions.loc[
            self._data.stimulus_conditions.stimulus_name == movie_type, "frame"
        ].max()

        # Get the number for the first frame of each second
        n_first_frames = np.arange(0, max_frame, 30).astype(int)

        # Select condition indices
        frame_id = self._data.stimulus_conditions.loc[
            self._data.stimulus_conditions.stimulus_name == movie_type
            ].index.values[n_first_frames[sec - 1]]

        # Get presentation indices
        repeat_idx = self._data.stimulus_presentations.loc[
            self._data.stimulus_presentations.stimulus_condition_id == frame_id
            ].index.values

        return repeat_idx

    def get_stim_spike_count(self, stim_type: str, roi: str):
        """
        Counts the number of spikes in response to repetitions of some stimulus.

        Parameters
        ----------
        stim_type : str
            Stimulus (drifting_grating, etx)

        roi : str
            Name of the brain region

        Returns
        -------
        pd.Dataframe
        """
        # Get indices
        repeat_idx = self._get_stim_idx(stim_type)
        roi_idx = self._get_roi_idx(roi)

        # Get the counts
        time_step = .001
        time_bins = np.arange(0, 0.5 + time_step, time_step)
        spikes = self._data.presentationwise_spike_counts(
            stimulus_presentation_ids=repeat_idx,
            unit_ids=roi_idx,
            bin_edges=time_bins
        )
        # Convert xarray to dataframe
        spikes = spikes.to_dataframe()

        # Add stimulus properties
        cols = ["start_time", "stop_time", "stimulus_block", "orientation", "temporal_frequency", "stimulus_name",
                "stimulus_condition_id"]
        spikes = pd.merge(
            spikes,
            self._data.stimulus_presentations[cols],
            left_on="stimulus_presentation_id",
            right_index=True
        )

        # Count the number of spikes in all time_relative_to_stimulus_onset s
        grp_cols = ["stimulus_presentation_id", "unit_id"] + cols
        df = spikes.groupby(grp_cols, as_index=False)["spike_counts"].sum()

        # Add info about the roi and which second of the movie
        df["roi"] = roi

        # Add running speed and pupil
        self._add_behavioral(df, repeat_idx)

        return df

    def get_movie_spike_count(self, movie_type: str, roi: str, sec: int):
        """
        Counts the number of spikes in response to a window of 1s of a movie.

        movie_type : str
        roi : str
        sec : int

        Returns
        -------
        pd.Dataframe
        """
        # Make bins
        time_step = 0.001
        time_bins = np.arange(0, 1 + time_step, time_step)

        # Get indices
        repeat_idx = self._get_movie_idx(movie_type, sec)
        roi_idx = self._get_roi_idx(roi)

        # Get the counts
        spikes = self._data.presentationwise_spike_counts(
            stimulus_presentation_ids=repeat_idx,
            unit_ids=roi_idx,
            bin_edges=time_bins
        )

        # Convert xarray to dataframe
        spikes = spikes.to_dataframe()

        # Add stuff to the dataframe
        cols = ["start_time", "stop_time", "stimulus_block", "stimulus_name", "stimulus_condition_id"]
        spikes = pd.merge(
            spikes, self._data.stimulus_presentations[cols],
            left_on="stimulus_presentation_id",
            right_index=True
        )
        spikes.reset_index(inplace=True)

        # Sum the number of spikes during 1s (over all time_relative_to_stimulus_onset times)
        grp_cols = ["stimulus_presentation_id", "unit_id"] + cols
        df = spikes.groupby(grp_cols, as_index=False)["spike_counts"].sum()

        # Add info about the roi and which second of the movie
        df["roi"] = roi
        df["movie_sec"] = sec

        # Add running speed
        self._add_behavioral(df, repeat_idx)

        return df

    def _add_behavioral(self, df, repeat_idx):
        """
        Adds running speed and pupil area to a dataframe based on time of stimulus presentation

        Parameters
        ----------
        df : pd.Dataframe
            Input dataframe to be modified

        repeat_idx : np.array
            Index of repetitions of the stimulus

        Returns
        -------
        None
        """
        # Add running speed
        run_data = self._data.running_speed
        for idx in repeat_idx:
            start = df.loc[df.stimulus_presentation_id == idx, "start_time"].mean()  # these are all the same number
            stop = start + 1
            median_speed = run_data.loc[
                (run_data["start_time"] > start) & (run_data["end_time"] < stop), "speed"
            ].median()
            mean_speed = run_data.loc[
                (run_data["start_time"] > start) & (run_data["end_time"] < stop), "speed"
            ].mean()
            df.loc[df.stimulus_presentation_id == idx, "median_speed"] = median_speed
            df.loc[df.stimulus_presentation_id == idx, "mean_speed"] = mean_speed

        # Add pupil area
        pupil_data = self._data.get_screen_gaze_data()[["raw_pupil_area"]]
        for idx in repeat_idx:
            start = df.loc[df.stimulus_presentation_id == idx, "start_time"].mean()
            stop = start + 1
            area = pupil_data.loc[
                (pupil_data.index > start) & (pupil_data.index < stop), "raw_pupil_area"
            ].mean()
            df.loc[df.stimulus_presentation_id == idx, "pupil_area"] = area

    def overall_pupil_run_average(self, bin_size: int):
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
        pupil_means = np.zeros((len(self._all_sessions), len(time_bins)))
        run_means = np.zeros((len(self._all_sessions), len(time_bins)))
        err = False

        for s, session in tqdm(enumerate(self._all_sessions)):

            # load session metadata
            _data = self.cache.get_session_data(session)

            # get pupil and running data
            try:
                pupil = _data.get_screen_gaze_data()[["raw_pupil_area"]]
                # Get the pupil mean
                pupil_mean = np.array(
                    [pupil.loc[(pupil.index > b) & (pupil.index < b + 1), "raw_pupil_area"].mean() for b in time_bins]
                )
            except TypeError:
                err = True
                print(f"Session {session} has no eye tracing data!")

            try:
                run = _data.running_speed
                run["speed"] = abs(run["velocity"])  # absolute value of speed
                # get running speed mean for each bin
                run_mean = np.array(
                    [run.loc[(run.start_time > b) & (run.end_time < b + 1), "speed"].mean() for b in time_bins]
                )
            except TypeError:
                err = True
                print(f"Session {session} has no running data")

            # save only if both exist
            if not err:
                run_means[s] = run_mean
                pupil_means[s] = pupil_mean

        return pupil_means, run_means




