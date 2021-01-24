import csv
import json

import numpy

import pickle

paths = {
	'datasets': "$HOME/Thesis/datasets",
	'harmonic_change_temp': "./out/harmonic_change/",
	'eval_temp': "./out/eval/",
	'chroma_temp': "./out/chroma/",
	'tonal_model_temp': "./out/tonal_model/",
	'gaussian_blur_temp': "./out/gaussian_blur/",
	'blur_temp': "./out/blur/",
	'audio_temp': "./out/audio/",
	'node_in': "in/node1.json"
}


def save_centroids(centroids, name_file):
	with open('out/' + name_file + '_centroids.csv', 'w') as f:
		f.write("c[0],c[1],c[2],c[3],c[4],c[5]\n")
		for j in range(0, centroids.shape[1]):
			for i in range(0, centroids.shape[0]):
				f.write("%s," % centroids[i][j])
			f.write("\n")


def save_hcdf(vector, changes, name_file):
	with open('out/' + name_file + '_hcdf.csv', 'w') as f:
		f.write("euclidian distance,\n")
		for s in vector:
			f.write("%s\n" % s)


def save_changes(changes, name_file):
	with open('out/' + name_file + '_changes.csv', 'w') as f:
		f.write("from time,to time,c[0],c[1],c[2],c[3],c[4],c[5]\n")
		for i in range(0, changes.shape[0]):
			for j in range(0, changes.shape[1]):
				f.write("%s," % changes[i][j])
			f.write("\n")


def save_results(centroid_vector, harmonic_function, changes, name_file):
	save_centroids(centroid_vector, name_file)
	save_hcdf(harmonic_function, changes, name_file)
	save_changes(changes, name_file)


def load_real_onset(filename, dataset="/Queen/"):
	real_chords = []
	delimeter = ' ' if dataset == '/The Beatles/' else '\t'
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=delimeter)

		for row in csv_reader:
			real_chords.append(float(row[0]))

	return numpy.array(real_chords)


def load_real_chords(filename, dataset="/Queen/"):
	real_chords = []
	delimeter = ' ' if dataset == '/The Beatles/' else '\t'
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=delimeter)

		for row in csv_reader:
			real_chords.append(str(row[0]) + "-" + str(row[2]))

	return numpy.array(real_chords)


# Functions for save informations in hcdf
def save_json(dictionary, name_file):
	with open(name_file, 'w') as fp:
		json.dump(dictionary, fp, sort_keys=True, indent=4)


def load_json(name_file):
	data = None
	with open(name_file, 'r') as fp:
		data = json.load(fp)
	return data


def save_binary(dictionary, name_file):
	with open(name_file, 'wb') as fp:
		pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_binary(name_file):
	data = None
	with open(name_file, 'rb') as fp:
		data = pickle.load(fp)
	return data


def get_name_audio(name_file, hpss, sr, codec='.pickle'):
	hpss_name = 'hpss' if hpss else 'none'
	return paths["audio_temp"] + "audio." + name_file + '.' + hpss_name + '.' + str(sr) + codec


def get_name_chromagram(name_file, hpss, chroma, codec='.pickle'):
	hpss_name = 'hpss' if hpss else 'none'
	return paths["chroma_temp"] + "chroma." + name_file + '.' + hpss_name + '.' + chroma + codec


def get_name_tonal_model(name_file, hpss, chroma, tonal_model, codec='.pickle'):
	hpss_name = 'hpss' if hpss else 'none'
	return paths["tonal_model_temp"] + "tonal." + name_file + '.' + hpss_name + '.' + chroma + '.' + tonal_model + codec


def get_name_gaussian_blur(name_file, hpss, chroma, tonal_model, blur, sigma, log_compresion, codec='.pickle'):
	hpss_name = 'hpss' if hpss else 'none'
	return paths["gaussian_blur_temp"] + "blur." + name_file + '.' + hpss_name + '.' + chroma + '.' + tonal_model + '.' + blur + '.' + 'sigma' + str(sigma) + '.' + log_compresion + codec


def get_name_eval(filename, hpss, tonal_model, chroma, blur, sigma, log_compresion, distance, codec='.json'):
	hpss_name = 'hpss' if hpss else 'none'
	return paths["eval_temp"] + "eval." + filename + '.' + hpss_name + '.' + chroma + '.' + tonal_model + '.' + blur+ '.' + 'sigma' + str(sigma) + '.' + log_compresion + '.' + distance + codec


def get_name_harmonic_change(filename, hpss, tonal_model, chroma, blur, sigma, log_compresion, distance, codec='.pickle'):
	hpss_name = 'hpss' if hpss else 'none'
	return paths["harmonic_change_temp"] + "hcdf." + filename + '.' + hpss_name + '.' + chroma + '.' + tonal_model + '.' + blur + '.' + 'sigma' + str(sigma) + '.' + log_compresion + '.' + distance + codec
