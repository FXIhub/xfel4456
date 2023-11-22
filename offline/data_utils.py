from multiprocessing import Pool
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
from tqdm.auto import tqdm
from extra_data import stack_detector_data, by_id, DataCollection
from extra_geom import JUNGFRAUGeometry

def load_trains(sel: DataCollection, trains: Sequence[int], config: Dict[str, Any], stacked: bool=True,
                assembled: bool=True, roi: Optional[Tuple[int, int, int, int]]=None) -> Dict[str, Any]:
    """Load train data for JUNGFRAU detector.

    Args:
        sel : Data collection from :func:`extra_data.open_run`.
        trains : A list of train indexes to load. You can retrieve a list of indexes with
            :code:`sel.train_ids`.
        config : Configuration dictionary. Contains the following parameters:

            - `data_type` : Choose between 'raw' and 'proc'.
            - `geom_path` : A path to the JUNGFRAU geometry ('.geom') file.
            - `modules` : Number of detector modules to load.
            - `pattern` : Regex to find the module number in source names. Should contain a group
              which can be converted to an integer. E.g. ``r'/DET/JNGFR(\\d+)'`` for
              one JUNGFRAU naming convention.
            - `sources` : glob patterns to select the JUNGFRAU detector sources.
            - `starts_at` : By default, uses module numbers starting at 0 (e.g. 0-15 inclusive).
              If the numbering is e.g. 1-16 instead, pass ``starts_at = 1``.

        stacked : Load stacked data from JUNGFRAU detector modules.
        assembled : Load assembled frame from JUNGFRAU detector modules.
        roi : Load only data in the region of interest `(y_min, y_max, x_min, x_max)` from
            assembled frames if provided.

    Returns:
        A dictionary with the following keywords:
        
        - 'stacked': Dictionary of stacked arrays of data from JUNGFRAU detector modules for each
          data key in `config['sources']`. If `stacked` is True.
        - 'assembled' : Dictionary of assembled frames from JUNGFRAU detector modules for each data
          key in `config['sources']`. If `stacked` is True.
    """
    if not assembled and not stacked:
        raise ValueError('Either assembled or stacked must be True')

    if assembled:
        if config['geom_path'] is None:
            raise ValueError('No geometry file has been provided')

        geom = JUNGFRAUGeometry.from_crystfel_geom(config['geom_path'])

    result = {}
    if stacked:
        result['stacked'] = {data_key: [] for _, data_key in config['sources']}
    if assembled:
        result['assembled'] = {data_key: [] for _, data_key in config['sources']}

    sel = sel.select_trains(by_id[trains])

    with tqdm(sel.trains(require_all=True), desc="Reading run", total=len(sel.train_ids),
              disable=len(sel.train_ids) <= 1) as pbar:
        for tid, train_data in pbar:
            pbar.set_postfix_str(f"reading train {tid}")
            for _, data_key in config['sources']:
                stacked_dset = stack_detector_data(train_data, data_key, modules=config['modules'],
                                                   starts_at=config['starts_at'], pattern=config['pattern'])
                stacked_dset = np.nan_to_num(stacked_dset, nan=0, posinf=0, neginf=0)

                if stacked:
                    result['stacked'][data_key].append(stacked_dset)

                if assembled:
                    assembled_dset = np.nan_to_num(geom.position_modules(stacked_dset)[0])
                    if roi is None:
                        result['assembled'][data_key].append(assembled_dset)
                    else:
                        result['assembled'][data_key].append(assembled_dset[roi[0]:roi[1], roi[2]:roi[3]])

    for data_type, data in result.items():
        for data_key, dset in data.items():
            result[data_type][data_key] = np.concatenate(dset, axis=0)

    return result

def initialise_worker(config: Dict[str, Any], stacked: bool, assembled: bool,
                      roi: Optional[Tuple[int, int, int, int]]):
    global worker
    geom = JUNGFRAUGeometry.from_crystfel_geom(config['geom_path'])

    def _worker(train_id: int, train_data: Dict[str, Any]) -> Tuple[int, Dict[str, Dict[str, np.ndarray]]]:
        train_dset = {}
        if stacked:
            train_dset['stacked'] = {}
        if assembled:
            train_dset['assembled'] = {}

        for _, data_key in config['sources']: 
            stacked_dset = stack_detector_data(train_data, data_key, modules=config['modules'],
                                               starts_at=config['starts_at'], pattern=config['pattern'])
            stacked_dset = np.nan_to_num(stacked_dset, nan=0, posinf=0, neginf=0)

            if stacked:
                train_dset['stacked'][data_key] = stacked_dset

            if assembled:
                assembled_dset = np.nan_to_num(geom.position_modules(stacked_dset)[0])
                if roi is None:
                    train_dset['assembled'][data_key] = assembled_dset
                else:
                    train_dset['assembled'][data_key] = assembled_dset[:, roi[0]:roi[1], roi[2]:roi[3]]

        return train_id, train_dset

    worker = _worker

def read_train(train: Tuple[int, Dict[str, np.ndarray]]) -> Tuple[int, Dict[str, Dict[str, np.ndarray]]]:
    train_id, train_data = train
    return worker(train_id, train_data)

def load_trains_pool(sel: DataCollection, trains: Sequence[int], config: Dict[str, Any], processes: int,
                     stacked: bool=True, assembled: bool=True, roi: Optional[Tuple[int, int, int, int]]=None) -> Dict[str, Any]:
    """Concurrently load train data for JUNGFRAU detector.

    Args:
        sel : Data collection from :func:`extra_data.open_run`.
        trains : A list of train indexes to load. You can retrieve a list of indexes with
            :code:`sel.train_ids`.
        config : Configuration dictionary. Contains the following parameters:

            - `data_type` : Choose between 'raw' and 'proc'.
            - `geom_path` : A path to the JUNGFRAU geometry ('.geom') file.
            - `modules` : Number of detector modules to load.
            - `pattern` : Regex to find the module number in source names. Should contain a group
              which can be converted to an integer. E.g. ``r'/DET/JNGFR(\\d+)'`` for
              one JUNGFRAU naming convention.
            - `sources` : glob patterns to select the JUNGFRAU detector sources.
            - `starts_at` : By default, uses module numbers starting at 0 (e.g. 0-15 inclusive).
              If the numbering is e.g. 1-16 instead, pass ``starts_at = 1``.

        processes : Number of concurrent processes.
        stacked : Load stacked data from JUNGFRAU detector modules.
        assembled : Load assembled frame from JUNGFRAU detector modules.
        roi : Load only data in the region of interest `(y_min, y_max, x_min, x_max)` from
            assembled frames if provided.

    Returns:
        A dictionary with the following keywords:
        
        - 'stacked': Dictionary of stacked arrays of data from JUNGFRAU detector modules for each
          data key in `config['sources']`. If `stacked` is True.
        - 'assembled' : Dictionary of assembled frames from JUNGFRAU detector modules for each data
          key in `config['sources']`. If `stacked` is True.
    """
    if not assembled and not stacked:
        raise ValueError('Either assembled or stacked must be True')

    if assembled:
        if config['geom_path'] is None:
            raise ValueError('No geometry file has been provided')

    result = {}
    if stacked:
        result['stacked'] = {data_key: [] for _, data_key in config['sources']}
    if assembled:
        result['assembled'] = {data_key: [] for _, data_key in config['sources']}

    sel = sel.select_trains(by_id[trains])

    with Pool(processes=processes, initializer=initialise_worker, initargs=(config, stacked, assembled, roi)) as pool:
        with tqdm(pool.imap(read_train, sel.trains(require_all=True)), desc="Reading run", total=len(sel.train_ids),
                  disable=len(sel.train_ids) <= 1) as pbar:
            for tid, train_dset in pbar:
                pbar.set_postfix_str(f"reading train {tid}")
                for data_type, data in train_dset.items():
                    for data_key, dset in data.items():
                        result[data_type][data_key].append(dset)

    for data_type, data in result.items():
        for data_key, dset in data.items():
            result[data_type][data_key] = np.concatenate(dset, axis=0)

    return result
