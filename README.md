# Harmonic Change Detection Function (HCDF) library

This library is used to compute HCDF. As described is descrived [here](). As many of the solutions overlap partilly, all the data computed in the different blocks of the algorithm are saved in the out folder in order to not compute same blocks parameterization two times.


## Installation
Install Vamp-plugins:
- (NNLSchroma) http://www.isophonics.net/nnls-chroma
- (HPCPchroma) http://mtg.upf.edu/technologies/hpcp
Install dependencies:
```BASH
	python3 setup.py
```

## Usage
The library can be imported as a module with `import HCDF`. All the functions than begins by get are blocks from HCDF
function. The rest are auxiliar functions.

HCDF.py act as script allowing the user to print to the console Harmonic Change Detection Function (HCDF)
focus performance on recall or f-score. It is assumed that the first command line argument is
the name file of the audio file located in audio_files and the second one is if is focus on 
recall or f-score.

### Example of use

With target on maximize f-score:

```BASH
python3 f-score file/name
```

With target on maximize recall:

```BASH
python3 recall file/name
```


## References

https://librosa.github.io
https://vamp-plugins.org
https://github.com/aframires/TIVlib