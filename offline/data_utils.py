from multiprocessing import Pool
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
from tqdm.auto import tqdm
from extra_data import stack_detector_data, by_id, DataCollection
from extra_geom import JUNGFRAUGeometry
from streak_finder.src import median, robust_mean, robust_lsq

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
          key in `config['sources']`. If `assembled` is True.
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
          key in `config['sources']`. If `assembled` is True.
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

    with Pool(processes=processes, initializer=initialise_worker,
              initargs=(config, stacked, assembled, roi)) as pool:
        with tqdm(pool.imap(read_train, sel.trains(require_all=True)), desc="Reading run",
                  total=len(sel.train_ids), disable=len(sel.train_ids) <= 1) as pbar:
            for tid, train_dset in pbar:
                pbar.set_postfix_str(f"reading train {tid}")
                for data_type, data in train_dset.items():
                    for data_key, dset in data.items():
                        result[data_type][data_key].append(dset)

    for data_type, data in result.items():
        for data_key, dset in data.items():
            result[data_type][data_key] = np.concatenate(dset, axis=0)

    return result

def generate_gaps(length: int, gap_size: int) -> np.ndarray:
    """Generate asic gaps for a given grid size `length` of the detector module.
    """
    gaps = np.append(np.arange(0, length, 258), length - 1)
    shift = -gap_size * np.ones(gaps.size, dtype=int)
    shift[0] = gap_size - 1
    shift[-1] = -gap_size + 1
    return np.sort(np.stack([gaps, gaps + shift], axis=1), axis=1)

def create_mask_assembled(assembled: np.ndarray, vmin: float, vmax: float, std_max: float) -> np.ndarray:
    """Mask out bad pixels. All pixels which mean value is above `vmax` and which
    standard deviation is above `std_max` are considered bad.

    Args:
        assembled : Assembled frames from JUNGFRAU detector.
        vmax : Mean value threshold.
        std_max : Standard deviation threshold.

    Returns:
        A bad pixel mask, where bad pixels are False.
    """
    mean_frame = np.mean(assembled, axis=0)
    std_frame = np.std(assembled, axis=0)

    mask = (mean_frame > vmin) & (mean_frame < vmax) & (std_frame < std_max)
    mask &= np.invert(np.isnan(mean_frame) | np.isinf(mean_frame))
    return mask

def create_mask_stacked(stacked: np.ndarray, vmin: float, vmax: float, std_max: float,
                        asic: bool=True, gap_size: int=3) -> np.ndarray:
    """Mask out bad pixels. All pixels which mean value is above `vmax` and which
    standard deviation is above `std_max` are considered bad.

    Args:
        stacked : Assembled frames from JUNGFRAU detector.
        vmax : Mean value threshold.
        std_max : Standard deviation threshold.
        asic : Mask the asic gaps if True.

    Returns:
        A bad pixel mask, where bad pixels are False.
    """
    mean_frame = np.mean(stacked, axis=0)
    std_frame = np.std(stacked, axis=0)

    mask = (mean_frame > vmin) & (mean_frame < vmax) & (std_frame < std_max)
    mask &= np.invert(np.isnan(mean_frame) | np.isinf(mean_frame))

    if asic:
        for begin, end in generate_gaps(stacked.shape[-1], gap_size):
            mask[..., begin:end] = False

        for begin, end in generate_gaps(stacked.shape[-2], gap_size):
            mask[..., begin:end, :] = False

    return mask

def create_whitefield(data: np.ndarray, mask: Optional[np.ndarray]=None, method: str='robust-mean',
                      num_threads: int=1) -> np.ndarray:
    """Generate a white-field from a stack of frames by using either median or robust mean.

    Args:
        data : Data from JUNGFRAU detector.
        mask : Bad pixel mask.
        method : Method used to calculate the white-field. Accepts the following options:

            * 'median' : Taking a median through a stack of frames.
            * 'robust-mean' : Calculating a robust mean through a stack of frames.

        num_threads : Number of threads used in calculations.

    Returns:
        White-field array.
    """
    if method == 'median':
        return median(inp=data, mask=mask, axis=0, num_threads=num_threads)
    if method == 'robust-mean':
        if mask is not None:
            data *= mask
        return robust_mean(inp=data, axis=0, num_threads=num_threads)
    raise ValueError(f'invalid method argument: {method}')

def calculate_snr(frames: np.ndarray, whitefield: np.ndarray, method: str='robust-lsq',
                      num_threads: int=1) -> np.ndarray:
    r"""Scale a given white-field (`whitefield`) to a stack of frames (`frames`) and
    calculate signal-to-noise ratio as :math:`SNR = \frac{I - W}{\sigma(I)}`.

    Args:
        frames : A stack of frames from JUNGFRAU detector.
        whitefield : White-field.
        method : Method used to find the scaling factors for each of the frames. Accepts
            the following options:

            * 'least-squares' : Apply ordinary least-squares.
            * 'robust-lsq' : Apply robust least-squares.

        num_threads : Number of threads used in calculations.

    Returns:
        A scaled stack of white-fields.
    """
    std = np.std(frames, axis=0)
    y = np.where(std, frames / std, 0.0)
    W = np.where(std, whitefield / std, 0.0)[None, ...]

    if method == 'robust-lsq':
        x = robust_lsq(W=W, y=y, axis=(1, 2), num_threads=num_threads)
    elif method == 'least-squares':
        x = np.mean(y * W, axis=(1, 2))[:, None] / np.mean(W * W, axis=(1, 2))
    else:
        raise ValueError(f"invalid method argument: {method}")

    whitefields = np.sum(x[..., None, None] * W * std, axis=1)
    return np.where(std, (frames - whitefields) / std, 0.0)
