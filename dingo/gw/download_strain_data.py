import numpy as np
import pycbc.psd
from gwpy.timeseries import TimeSeries

from dingo.gw.domains import FrequencyDomain
from dingo.gw.gwutils import (
    get_window,
    get_window_factor,
)


def estimate_single_psd(
    det,
    time_start,
    time_segment,
    window,
    f_s=4096,
    num_segments: int = 128,
    channel=None,
):
    """
    Download strain data and generate a PSD based on these. Use num_segments of length
    time_segment, starting at GPS time time_start.

    Parameters
    ----------
    det: str
        detector
    time_start: float
        start GPS time for PSD estimation
    time_segment: float
        time for a single segment for PSD information, in seconds
    window: Union(np.ndarray, dict)
        Window used for PSD generation, needs to be the same as used for Fourier
        transform of event strain data.
        Either provided directly as np.ndarray, or as dict in which case the window is
        generated by window = dingo.gw.gwutils.get_window(**window).
    num_segments: int = 256
        number of segments used for PSD generation
    channel: str
        If provided, will download the data from the channel instead of gwosc using TimeSeries.get()
    Returns
    -------
    psd: np.array
        array of psd
    """
    # download strain data for psd
    time_end = time_start + time_segment * num_segments
    if channel:
        psd_strain = TimeSeries.get(channel, time_start, time_end)
        # TODO: We currently assume that sample rate of channel matches that provided in the settings?
    else:
        psd_strain = TimeSeries.fetch_open_data(
            det, time_start, time_end, sample_rate=f_s, cache=False
        )
    psd_strain = psd_strain.to_pycbc()

    # optionally generate window
    if type(window) == dict:
        window = get_window(**window)
    assert (
        len(window) == len(psd_strain) / num_segments
    ), "Window does not match strain. Is sampling frequency f_s off?"

    # generate PSD from strain data
    psd = pycbc.psd.estimate.welch(
        psd_strain,
        seg_len=len(window),
        seg_stride=len(window),
        window=window,
        avg_method="median",
    )

    return np.array(psd)


def download_strain_data_in_FD(det, time_event, time_segment, time_buffer, window):
    """
    Download strain data for a GW event at GPS time time_event. The segment is
    time_segment seconds long, including time_buffer seconds after the event. The
    strain is Fourier transformed, the frequency domain strain is then time shifted by
    time_buffer, such that the event occurs at t=0.

    Parameters
    ----------
    det: str
        detector
    time_event: float
        GPS time of the event
    time_segment: float
        length of the strain segment, in seconds
    time_buffer: float
        specifies buffer time after time_event, in seconds
    window: Union(np.ndarray, dict)
        Window used for Fourier transforming the event strain data.
        Either provided directly as np.ndarray, or as dict in which case the window is
        generated by window = dingo.gw.gwutils.get_window(**window).


    Returns
    -------
    event_strain: np.array
        array with the frequency domain strain
    """
    # download strain data
    print("Downloading strain data for event.", end=" ")
    event_strain = TimeSeries.fetch_open_data(
        det,
        time_event + time_buffer - time_segment,
        time_event + time_buffer,
        cache=True,
    )
    print("Done.")

    # transform to FD
    if type(window) == dict:
        window = get_window(**window)
    assert len(window) == len(
        event_strain
    ), "Window does not match strain. Is sampling frequency f_s off?"
    event_strain = event_strain.to_pycbc()
    event_strain = (event_strain * window).to_frequencyseries()

    # time shift by time_buffer, such that event happens at time 0
    event_strain = event_strain.cyclic_time_shift(time_buffer)

    return np.array(event_strain)


def download_event_data_in_FD(
    detectors,
    time_event,
    time_segment,
    time_buffer,
    window,
    num_segments_psd=128,
):
    """
    Download event data in frequency domain. This includes strain data for the event at
    GPS time t_event as well as the correcponding ASD.

    Parameters
    ----------
    detectors: list
        list of detectors specified via strings
    time_event: float
        GPS time of the event
    time_segment: float
        length of the strain segment, in seconds
    time_buffer: float
        specifies buffer time after time_event, in seconds
    window: Union(np.ndarray, dict)
        Window used for Fourier transforming the event strain data.
        Either provided directly as np.ndarray, or as dict in which case the window is
        generated by window = dingo.gw.gwutils.get_window(**window).
    num_segments_psd: int = 128
        number of segments used for PSD generation

    Returns
    -------
    """
    data = {"waveform": {}, "asds": {}}
    for det in detectors:
        print("Detector {:}:".format(det))

        data["waveform"][det] = download_strain_data_in_FD(
            det, time_event, time_segment, time_buffer, window
        )
        data["asds"][det] = (
            estimate_single_psd(
                det,
                time_event + time_buffer - time_segment * (num_segments_psd + 1),
                time_segment,
                window,
                num_segments_psd,
            )
            ** 0.5
        )

    # build domain object
    f_s = len(data["waveform"][detectors[0]]) / time_segment
    domain = FrequencyDomain(
        f_min=0,
        f_max=f_s / 2,
        delta_f=1 / time_segment,
        window_factor=get_window_factor(window),
    )

    return data, domain
