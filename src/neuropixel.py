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

import numpy as np
import pandas as pd

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

        # Initiate parameters
        self._sessions = list()
        self._data = dict()

    @property
    def cache(self):
        """ Getter for experiment cache"""
        return self._cache

    def get_count_tables(self):
        """
        Collects all the relevant parts of a data and puts them in a pandas DataFrame
        """

    def load_sessions(self, session_id: Union[int, list, str]):
        """
        Loads neuropixel session data. The session_id can be specified as:

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
            self._sessions.append(session_id)
            self._data[session_id] = self.cache.get_session_data(session_id)

        # Load for a list of sessions
        elif isinstance(session_id, list):
            for sess in session_id:
                self._sessions.append(sess)
                self._data[sess] = self.cache.get_session_data(sess)

        # Load for a session type
        elif session_id in ["brain_observatory_1.1", "functional_connectivity"]:

            # select session type
            _sessions = sessions[sessions.session_type == session_id]

            # load individual sessions in the table
            for sess, row in _sessions.iterrows():

                # to check for bad files
                truncated_file = True
                directory = self.DATA_DIR / f'session_{str(sess)}'

                # load the data
                while truncated_file:
                    session = self.cache.get_session_data(sess)
                    try:
                        print(session.specimen_name)
                        truncated_file = False
                    except OSError:
                        shutil.rmtree(directory)
                        print(" Truncated spikes file, re-downloading")

        else:
            raise ValueError("Invalid session id(s).")

    def get_stim_idx(self, stim_type: str, condition: str):
        """

        Parameters
        ----------
        stim_type
        condition

        Returns
        -------

        """