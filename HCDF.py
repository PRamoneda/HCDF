"""HCDF python implementation

Author: Pedro Ramoneda Franco 
Year: 2020

This script allows the user to print to the console Harmonic Change Detection Function (HCDF)
focus performance on recall or f-score. It is assumed that the first command line argument is
the name file of the audio file located in audio_files and the second one is if is focus on 
recall or f-score.


This tool accepts comma separated value files (.csv) as well as excel
(.xls, .xlsx) files.

This script requires that `setup.py` requeriments be installed within the Python
environment you are running this script in. More over it is a need instal vamp plugins 
NNLS and HPCP

This file can also be imported as a module. All the functions than begins by get are blocks from HCDF
function. The rest are auxiliar.
	
"""
import os
from os import path

import sys
from TIVlib import TIV
import librosa
import numpy

numpy.set_printoptions(threshold=sys.maxsize)

from librosa.feature import chroma_cqt, tonnetz, chroma_cens, chroma_stft
from librosa.filters import get_window
from scipy.ndimage.filters import gaussian_filter
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.spatial.distance import cosine

from iotxt import load_real_onset, load_binary, get_name_harmonic_change, save_binary, get_name_chromagram, \
get_name_tonal_model, get_name_gaussian_blur, get_name_audio
import vamp


def get_distance(centroids, dist):
    """
    Returns the quantity of centroids per second

    Parameters
    ----------
    centroids : list of floats
        The file location of the spreadsheet
    sr : bool
        A flag used to print the columns to the console (default is False)

    Returns
    -------
    float
        centroids per second
    """
    ans = [0]
    if dist == 'euclidean':
        for j in range(1, centroids.shape[1] - 1):
            sum = 0
            for i in range(0, centroids.shape[0]):
                sum += ((centroids[i][j + 1] - centroids[i][j - 1]) ** 2)
            sum = numpy.math.sqrt(sum)

            ans.append(sum)

    if dist == 'cosine':
        for j in range(1, centroids.shape[1] - 1):
            distance_computed = cosine(centroids[:, j - 1], centroids[:, j + 1])
            ans.append(distance_computed)
    ans.append(0)

    return numpy.array(ans)


def centroids_per_second(y, sr, centroids):
    """
    Returns the quantity of centroids per second

    Parameters
    ----------
    y : list of floats
        The file location of the spreadsheet
    sr : bool
        A flag used to print the columns to the console (default is False)

    Returns
    -------
    float
        centroids per second
    """
    return sr * centroids.shape[1] / y.shape[0]


def get_peaks_hcdf(hcdf_function, c, threshold, rate_centroids_second, centroids):
    changes = [0]
    centroid_changes = [[centroids[j][0] for j in range(0, c.shape[0])]]
    last = 0
    for i in range(2, hcdf_function.shape[0] - 1):
        if hcdf_function[i - 1] < hcdf_function[i] and hcdf_function[i + 1] < hcdf_function[i]:
            centroid_changes.append([numpy.median(centroids[j][last + 1:i - 1]) for j in range(0, c.shape[0])])
            changes.append(i / rate_centroids_second)
            last = i
    return numpy.array(changes), centroid_changes


def everything_is_zero(vector):
    """Returns true if all the values of the vector are 0 if not return false

    Parameters
    ----------
    vector : list
        vector of reals

    Returns
    -------
    bool
        true or false depending if everything is 0 or not
    """
    for element in vector:
        if element != 0:
            return False
    return True


def complex_to_vector(vector):
    """transforms an array of i complex numbers in an array of 2*i elements where
        odd indexes are the real part and even indexes are the imaginary part.

        Parameters
        ----------
        vector : list
            list of complex numbers

        Returns
        -------
        list
            list of real numbers with odd indexes as the real part and even indexes as the imaginary part
    """
    ans = []
    for i in range(0, vector.shape[1]):
        row1 = []
        row2 = []
        for j in range(0, vector.shape[0]):
            row1.append(vector[j][i].real)
            row2.append(vector[j][i].imag)
        ans.append(row1)
        ans.append(row2)
    return numpy.array(ans)


def get_parameters_chroma(txt):
    """
        returns parameters of json "chroma-samplerate-framesize-overlap"

        Parameters
        ----------
        txt : str
            chroma-samplerate-framesize-overlap

        Returns
        -------
        dictionary with keys: {chroma, samplerate, framesize, overlap}
    """
    rows = txt.split("-")
    return {"chroma": rows[0], "sr": int(rows[1]), "fr": int(rows[2]), "off": int(rows[2]) // int(rows[3])}


def tonal_interval_space(chroma, symbolic=False):
    """
        returns tonal interval space from a vector of chromagrams

        Parameters
        ----------
        chroma : list
            list of chromagrams

        symbolic: bool
            True for symbolic musical audio tonal interval space and False for musical audio aproach

        Returns
        -------
        list of tonal interval space vectors
    """
    centroid_vector = []
    for i in range(0, chroma.shape[1]):
        each_chroma = [chroma[j][i] for j in range(0, chroma.shape[0])]
        # print(each_chroma)
        if everything_is_zero(each_chroma):
            centroid = [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]
        else:
            tonal = TIV.from_pcp(each_chroma, symbolic)
            centroid = tonal.get_vector()
        centroid_vector.append(centroid)
    return complex_to_vector(numpy.array(centroid_vector))


def check_parameters(chroma, blur, tonal_model, log_compresion, dist):
    chroma = get_parameters_chroma(chroma)["chroma"]
    chroma_type = {'nnls', 'hpcp', 'cqt', 'crp', 'stft', 'cens'}
    assert chroma in chroma_type, "Type of chroma is not correct ['nnls', 'hpcp', 'cqt', 'cens', 'stft']"
    blur_type = {'none', '17-points', 'full'}
    assert blur in blur_type, "Type of blur is not correct ['none', '17points', 'full']"
    tonal_model_type = {'tonnetz', 'TIV2', 'TIV2_symb', 'without_tc'}
    assert tonal_model in tonal_model_type, "Type of tonal model is not correct ['tonnetz', 'TIV2', 'TIV2_symb']"
    log_compresion_type = {'after', 'before', 'none'}
    assert log_compresion in log_compresion_type, "Type of log_compresion is not correct ['after', 'before', 'none']"
    distance_type = {'euclidean', 'cosine'}
    assert dist in distance_type, "Type of distance is not correct ['euclidian', 'cosine']"


def get_nnls(y, sr, fr, off):
    """
        returns nnls chromagram

        Parameters
        ----------
        y : number > 0 [scalar]
            audio

        sr: number > 0 [scalar]
            chroma-samplerate-framesize-overlap

        fr: number [scalar]
            frame size of windos

        off: number [scalar]
            overlap

        Returns
        -------
        list of chromagrams
    """
    plugin = 'nnls-chroma:nnls-chroma'
    chroma = list(vamp.process_audio(y, sr, plugin, output="chroma", block_size=fr, step_size=off))
    doce_bins_tuned_chroma = []
    for c in chroma:
        doce_bins_tuned_chroma.append(c['values'].tolist())
    return numpy.array(doce_bins_tuned_chroma).transpose()


def get_chromagram(y, sr, chroma):
    """
        returns chromagram

        Parameters
        ----------
        y : number > 0 [scalar]
            audio

        sr: number > 0 [scalar]
            target sampling rate

        chroma: str
            chroma-samplerate-framesize-overlap


        Returns
        -------
        list of chromagrams
    """
    params = get_parameters_chroma(chroma)
    chroma = params["chroma"]
    doce_bins_tuned_chroma = None
    if chroma == 'nnls':
        doce_bins_tuned_chroma = get_nnls(y, params["sr"], params["fr"], params["off"])
    elif chroma == 'cqt':
        win = get_window('blackmanharris', params["fr"])
        doce_bins_tuned_chroma = chroma_cqt(y=y, sr=params["sr"],
                                            C=None,
                                            hop_length=params["off"],
                                            norm=None,
                                            # threshold=0.0,
                                            window=win,
                                            fmin=110,
                                            n_chroma=12,
                                            n_octaves=4 if params["chroma"] == "cqt" and params["sr"] == 5525 else 5,
                                            bins_per_octave=36)
    elif chroma == 'cens':
        win = get_window('blackmanharris', params["fr"])
        doce_bins_tuned_chroma = chroma_cens(y=y, sr=params["sr"],
                                             C=None,
                                             hop_length=params["off"],
                                             norm=None,
                                             window=win,
                                             fmin=110,
                                             n_chroma=12,
                                             n_octaves=5,
                                             bins_per_octave=36)
    elif chroma == 'stft':
        win = get_window('blackmanharris', params["fr"])
        doce_bins_tuned_chroma = chroma_stft(y=y, sr=params["sr"], hop_length=params["off"], norm=None, window=win,
                                             n_chroma=12)
    return doce_bins_tuned_chroma


def chromagram(hpss, name_file, y, sr, chroma):
    """
        wrapper of get_chromagram for save all results for future same calculations

        Parameters
        ----------
        hpss : bool
            true or false depends on hpss block

        name_file: str
            name of the file that is being computed

        y : number > 0 [scalar]
            audio

        sr: number > 0 [scalar]
            target sampling rate

        chroma: str
            chroma-samplerate-framesize-overlap


        Returns
        -------
        list of chromagrams
    """
    name_chromagram = get_name_chromagram(name_file, hpss, chroma)
    if path.exists(name_chromagram):
        dic = load_binary(name_chromagram)
    else:
        # if mutex_global.mutex is not None:
        #     mutex_global.mutex.acquire()
        doce_bins_tuned_chroma = get_chromagram(y, sr, chroma)
        # if mutex_global.mutex is not None:
        #     mutex_global.mutex.release()
        dic = {'doce_bins_tuned_chroma': doce_bins_tuned_chroma}
        # dic_save = {'doce_bins_tuned_chroma': doce_bins_tuned_chroma.tolist()}
        save_binary(dic, name_chromagram)
    # save_json(dic_save, name_chromagram + '.json')
    return dic['doce_bins_tuned_chroma']


def get_tonal_centroid_transform(y, sr, tonal_model, doce_bins_tuned_chroma):
    """
        returns centroids from tonal model

        Parameters
        ----------
        hpss : bool
            true or false depends on hpss block

        name_file: str
            name of the file that is being computed

        y : number > 0 [scalar]
            audio

        sr: number > 0 [scalar]
            target sampling rate

        chroma: str
            chroma-samplerate-framesize-overlap

        tonal_model: str optional
            Tonal model block type. "TIV2" for Tonal Interval space focus on audio. "TIV2" for audio. "TIV2_Symb" for symbolic data.
            "tonnetz" for harte centroids aproach. Default TIV2\

        doce_bins_tuned_chroma: list
            list of chroma vectors

        Returns
        -------
        list of tonal centroids vectors
    """
    centroid_vector = None
    if tonal_model == 'tonnetz':
        centroid_vector = tonnetz(y=y, sr=sr, chroma=doce_bins_tuned_chroma)
    elif tonal_model == 'TIV2':
        centroid_vector = tonal_interval_space(doce_bins_tuned_chroma)
    elif tonal_model == 'TIV2_symb':
        centroid_vector = tonal_interval_space(doce_bins_tuned_chroma, symbolic=True)
    return centroid_vector


def tonal_centroid_transform(hpss, chroma, name_file, y, sr, tonal_model, doce_bins_tuned_chroma):
    """
        wrapper of tonal centroid transform for save all results for future same calculations

        Parameters
        ----------
        hpss : bool
            true or false depends on hpss block

        name_file: str
            name of the file that is being computed

        y : number > 0 [scalar]
            audio

        sr: number > 0 [scalar]
            target sampling rate

        chroma: str
            chroma-samplerate-framesize-overlap

        tonal_model: str optional
            Tonal model block type. "TIV2" for Tonal Interval space focus on audio. "TIV2" for audio. "TIV2_Symb" for symbolic data.
            "tonnetz" for harte centroids aproach. Default TIV2\

        doce_bins_tuned_chroma: list
            list of chroma vectors

        Returns
        -------
        list of tonal centroids vectors
    """
    name_tonal_model = get_name_tonal_model(name_file, hpss, chroma, tonal_model)
    if tonal_model == 'without_tc':
        dic = {'centroid_vector': doce_bins_tuned_chroma}
    else:
        if path.exists(name_tonal_model):
            dic = load_binary(name_tonal_model)
        else:
            centroid_vector = get_tonal_centroid_transform(y, sr, tonal_model, doce_bins_tuned_chroma)
            dic = {'centroid_vector': centroid_vector}
            save_binary(dic, name_tonal_model)
    return dic['centroid_vector']


def get_gaussian_blur(centroid_vector, blur, sigma):
    """
        Apply gaussian smoothing to tonal model centroids

        Parameters
        ----------
        centoid_vector: list
        tonal centroids of the tonal model

        sigma: number (scalar > 0) optional
        sigma of gaussian smoothing value. Default 11

        Returns
        -------
        list
        centroids blurred by gassuian smoothing
    """
    if blur == 'full':
        centroid_vector = gaussian_filter(centroid_vector, sigma=sigma)
    elif blur == '17-points':
        gauss_kernel = Gaussian1DKernel(17)
        i = 0
        for centroid in centroid_vector:
            centroid = convolve(centroid, gauss_kernel)
            centroid_vector[i] = centroid
    return numpy.array(centroid_vector)


def gaussian_blur(hpss, chroma, tonal_model, name_file, centroid_vector, log_compresion, blur, sigma):
    """
        Wrapper of get_gaussian_blur for save all results for future same calculations. If parameterization
        have been computed before get_gaussian_blur is not computed.

        Parameters
        ----------
        name_file: str
            name of the file that is being computed

        hpss: bool optional
            true or false depends is harmonic percussive source separation (hpss) block wants to be computed. Default False.

        sr: number > 0 [scalar]
            target sampling rate

        chroma: str optional
            "chroma-samplerate-framesize-overlap"
            chroma can be "CQT","NNLS", "STFT", "CENS" or "HPCP"
            samplerate as a number scalar
            frame size as a number scalar
            overlap number that a windows is divided

        tonal_model: str optional
            Tonal model block type. "TIV2" for Tonal Interval space focus on audio. "TIV2" for audio. "TIV2_Symb" for symbolic data.
            "tonnetz" for harte centroids aproach. Default TIV2

        centoid_vector: list
            tonal centroids of the tonal model

        sigma: number (scalar > 0) optional
            sigma of gaussian smoothing value. Default 11


        Returns
        -------
        list
            sample of audio
    """
    gaussian_blur = get_name_gaussian_blur(name_file, hpss, chroma, tonal_model, blur, sigma, log_compresion)
    if path.exists(gaussian_blur):
        dic = load_binary(gaussian_blur)
    else:
        centroid_vector = get_gaussian_blur(centroid_vector, blur, sigma)
        dic = {'centroid_vector': centroid_vector}
        # dic_save = {'centroid_vector': centroid_vector.tolist()}
        save_binary(dic, gaussian_blur)
    # save_json(dic_save, gaussian_blur + '.json')
    return dic['centroid_vector']


def get_audio(filename, hpss, sr):
    """
        Get audio as list

        Parameters
        ----------
        filename: str
                name of the file that is being computed witout format extension

        hpss: bool optional
            true or false depends is harmonic percussive source separation (hpss) block wants to be computed. Default False.

        sr: number > 0 [scalar]
            target sampling rate

        Returns
        -------
        list
            sample of audio
    """
    y, sr = librosa.load(filename, sr=sr, mono=True)
    if hpss:
        y = librosa.effects.harmonic(y)
    return y, sr


def audio(filename, name_file, hpss, sr):
    """
        Wrapper of get audio for save all results for future same calculations. If parameterization
        have been computed before get audio is not computed.

        Parameters
        ----------
        filename: str
            name of the file that is being computed witout format extension

        name_file: str
        name of the file that is being computed

        hpss: bool optional
        true or false depends is harmonic percussive source separation (hpss) block wants to be computed. Default False.

        sr: number > 0 [scalar]
        target sampling rate

        Returns
        -------
        list
        sample of audio
    """
    name_audio = get_name_audio(name_file, hpss, sr)
    if path.exists(name_audio):
        dic = load_binary(name_audio)
    else:
        y, sr = get_audio(filename, hpss, sr)
        dic = {'y': y, 'sr': sr}
        # dic_save = {'y': y.tolist(), 'sr': sr}
        save_binary(dic, name_audio)
    # save_json(dic_save, name_audio + '.json')
    return dic['y'], dic['sr']


def get_harmonic_change(filename: str, name_file: str, hpss: bool = False, tonal_model: str = 'TIV2',
                        chroma: str = 'cqt',
                        blur: str = 'full', sigma: int = 11, log_compresion: str = 'none', dist: str = 'euclidean'):
    """
        Computes Harmonic Change Detection Function

        Parameters
        ----------
        filename: str
                name of the file that is being computed witout format extension

        name_file: str
            name of the file that is being computed

        hpss : bool optional
            true or false depends is harmonic percussive source separation (hpss) block wants to be computed. Default False.

        tonal_model: str optional
            Tonal model block type. "TIV2" for Tonal Interval space focus on audio. "TIV2" for audio. "TIV2_Symb" for symbolic data.
            "tonnetz" for harte centroids aproach. Default TIV2

        chroma: str optional
            "chroma-samplerate-framesize-overlap"
            chroma can be "CQT","NNLS", "STFT", "CENS" or "HPCP"
            samplerate as a number scalar
            frame size as a number scalar
            overlap number that a windows is divided

        sigma: number (scalar > 0) optional
            sigma of gaussian smoothing value. Default 11

        distance: str optional
            type of distance measure used. Types can be "euclidean" for euclidean distance and "cosine" for cosine distance. Default "euclidean".


        Returns
        -------
        list
            harmonic changes (the peaks) on the song detected
        list
            HCDF function values
        number
            windows size
    """
    # audio
    y, sr = audio(filename, name_file, hpss, get_parameters_chroma(chroma)["sr"])

    # chroma
    doce_bins_tuned_chroma = chromagram(hpss, name_file, y, sr, chroma)

    # tonal_model
    centroid_vector = tonal_centroid_transform(hpss, chroma, name_file, y, sr, tonal_model, doce_bins_tuned_chroma)

    # blur
    centroid_vector_blurred = gaussian_blur(hpss, chroma, tonal_model, name_file, centroid_vector, log_compresion, blur,
                                            sigma)

    # harmonic distance and calculate peaks
    harmonic_function = get_distance(centroid_vector_blurred, dist)
    windows_size = centroids_per_second(y, sr, centroid_vector_blurred)
    changes, centroid_changes = get_peaks_hcdf(harmonic_function, centroid_vector_blurred, 0, windows_size,
                                               centroid_vector)

    return changes, harmonic_function, windows_size, numpy.array(centroid_changes)


def harmonic_change(filename: str, name_file: str, hpss: bool = False, tonal_model: str = 'TIV2', chroma: str = 'cqt',
                    blur: str = 'full', sigma: int = 11, log_compresion: str = 'none', distance: str = 'euclidean'):
    """
        Wrapper of harmonic change detection function for save all results for future same calculations. If parameterization
        have been computed before HCDF is not computed.

        Parameters
        ----------
        filename: str
                name of the file that is being computed witout format extension

        name_file: str
            name of the file that is being computed

        hpss : bool optional
            true or false depends is harmonic percussive source separation (hpss) block wants to be computed. Default False.

        tonal_model: str optional
            Tonal model block type. "TIV2" for Tonal Interval space focus on audio. "TIV2" for audio. "TIV2_Symb" for symbolic data.
            "tonnetz" for harte centroids aproach. Default TIV2

        chroma: str optional
            "chroma-samplerate-framesize-overlap"
            chroma can be "CQT","NNLS", "STFT", "CENS" or "HPCP"
            samplerate as a number scalar
            frame size as a number scalar
            overlap number that a windows is divided

        sigma: number (scalar > 0) optional
            sigma of gaussian smoothing value. Default 11

        distance: str optional
            type of distance measure used. Types can be "euclidean" for euclidean distance and "cosine" for cosine distance. Default "euclidean".


        Returns
        -------
        list
            harmonic changes (the peaks) on the song detected
        list
            HCDF function values
        number
            windows size
    """
    centroid_changes = []
    check_parameters(chroma, blur, tonal_model, log_compresion, distance)

    name_harmonic_change = get_name_harmonic_change(name_file, hpss, tonal_model, chroma, blur, sigma, log_compresion,
                                                    distance)
    if path.exists(name_harmonic_change):
        dic = load_binary(name_harmonic_change)
    else:
        changes, harmonic_function, windows_size, centroid_changes = get_harmonic_change(filename, name_file, hpss,
                                                                                         tonal_model, chroma,
                                                                                         blur, sigma, log_compresion,
                                                                                         distance)
        dic = {'changes': changes, 'harmonic_function': harmonic_function, 'windows_size': windows_size}

        save_binary(dic, name_harmonic_change)
    return dic['changes'], dic['harmonic_function'], dic['windows_size']


def main(argv):
    """ This program computes HCDF function focus performance on recall or precision

        Arguments
        ----------
        first one :
            recall or f-score
        second one :
           The name file of the audio file located in audio_files

        PRINTS
        -------
            a list of harmonic changes (the peaks) on the song detected
            a list with the HCDF function
            windows size

        Typical use
        Harmonical use

    """
    absolute_path = "./audio_files/"
    # file = "07_-_Please_Please_Me.wav"
    file = argv[2]
    if argv[1] == "f-score":
        print(harmonic_change(absolute_path + file,
                              file,
                              chroma='nnls-8000-1024-2',
                              hpss=True,
                              tonal_model='TIV2',
                              blur='full',
                              sigma=5,
                              distance='euclidean'
                              ))

    elif argv[1] == "recall":
        print(harmonic_change(absolute_path + file,
                              file,
                              chroma='stft-44100-2048-4',
                              hpss=True,
                              tonal_model='TIV2',
                              blur='full',
                              sigma=17,
                              distance='euclidean'))


if __name__ == '__main__':
    main(sys.argv)
