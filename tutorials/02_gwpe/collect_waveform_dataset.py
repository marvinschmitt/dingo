"""
Generate waveform dataset

Step 5

Load parameters, compressed polarizations, and SVD basis from .npy files
Save consolidated waveform dataset as HDF5.
"""
import argparse
import os
import yaml

import h5py
import numpy as np
from tqdm import tqdm

from dingo.api import setup_logger, logger
from .build_SVD_basis import find_chunk_number


def consolidate_dataset(num_chunks: int, basis_file: str,
                        parameters_file: str, dataset_file: str,
                        alpha=None, single_precision=False):
    """
    Load data files and output a consolidated dataset in HDF5 format.

    Parameters
    ----------
    num_chunks:
        Number of polarization data chunks the full parameter array was split into
    basis_file:
        .npy file containing the polarization SVD basis matrix V
    parameters_file:
        .npy file containing the structured parameter array
    dataset_file:
        Output HDF5 file for the dataset
    alpha:
        Rescale SVD coefficients of waveforms by this factor
    """
    logger.info('Load polarization data for all chunks ...')
    pol_keys = ['h_plus', 'h_cross']
    pol_data = {k: np.vstack([np.load(f'{k}_coeff_{idx}.npy')
                             for idx in tqdm(np.arange(num_chunks))])
                for k in pol_keys}

    if single_precision:
        dtype = np.float32
        logger.info(f'Using single precision.')
    else:
        dtype = np.float64
        logger.info(f'Using double precision.')

    logger.info('Saving dataset to HDF5 ...')
    fp = h5py.File(dataset_file, 'w')

    if alpha is not None:
        logger.info(f'Rescaled SVD coefficients by factor {alpha}.')
    else:
        alpha = 1.0
    fp.attrs['Rescaled SVD coefficients by factor'] = alpha

    # Polarization projection coefficients
    grp = fp.create_group('waveform_polarizations')
    logger.info('waveform_polarizations  Group:')
    for k, v in pol_data.items():
        grp.create_dataset(str(k), data=v*alpha, dtype=dtype)
        logger.info(f'\t{k}: {v.shape}')

    # Parameter samples
    parameters = np.load(parameters_file)
    logger.info(f'parameters              Dataset ({len(parameters)}, {len(parameters.dtype)})')
    fp.create_dataset('parameters', data=parameters)

    # SVD polarization basis
    basis_V_matrix = np.load(basis_file)
    logger.info(f'rb_matrix_V             Dataset {basis_V_matrix.shape}')
    fp.create_dataset('rb_matrix_V', data=basis_V_matrix)

    with open('settings.yaml', 'r') as fp:
        settings = yaml.safe_load(fp)
    fp.attrs['settings'] = ''.join(f'{k}: {v}' for k, v in settings.items())

    fp.close()
    logger.info('Done')


def main():
    parser = argparse.ArgumentParser(description="""
        Collect compressed waveform polarizations and parameters.
        Save consolidated waveform dataset in HDF5 format.
    """)
    parser.add_argument('--waveforms_directory', type=str, required=True,
                        help='Directory containing waveform data, basis, and parameter file.')
    parser.add_argument('--parameters_file', type=str, required=True,
                        help='Parameter file for compressed production waveforms.')
    parser.add_argument('--basis_file', type=str, default='polarization_basis.npy')
    parser.add_argument('--settings_file', type=str, default='settings.yaml')
    parser.add_argument('--dataset_file', type=str, default='waveform_dataset.hdf5')
    parser.add_argument('--rescale_coefficients_factor', type=float, default=1.0e21)
    parser.add_argument('--single_precision', default=False, action='store_true')
    args = parser.parse_args()


    os.chdir(args.waveforms_directory)
    setup_logger(outdir='.', label='collect_waveform_dataset', log_level="INFO")
    logger.info('Executing collect_waveform_dataset:')

    num_chunks, chunk_size = find_chunk_number(args.parameters_file, compressed=True)
    consolidate_dataset(num_chunks, args.basis_file, args.parameters_file,
                        args.dataset_file, args.rescale_coefficients_factor,
                        args.single_precision)


if __name__ == "__main__":
    main()

