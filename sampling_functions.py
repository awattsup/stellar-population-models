import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.table import Table






def sample_IMF_ratio(slope = -3.35, m_upper = 12, m_lower = 1):

	base_slope = -1.95
	base_m_upper = 120

	slope += 1
	base_slope += 1

	mass_range = np.arange(1,120,0.05)

	base_prob = (mass_range**base_slope - m_lower**base_slope) / (base_m_upper**base_slope - m_lower**base_slope)
	IMF_prob = (mass_range**slope - m_lower**slope) / (m_upper**slope - m_lower**slope)

	# plt.plot(mass_range,prob)
	# plt.plot(mass_range,base_prob)

	prob_ratio = (1- prob) / base_prob

	# plt.plot(mass_range, prob_ratio)

	keep = []
	for ii in range(len(masses)):
		prob = np.random()
		if prob < np.interp(masses[ii],mass_range,prob_ratio):
			keep.extend([ii])

	return keep



def sample_IMF(slope=-2.35, m_lower = 1, m_upper = 120, Nstar = 1000):

	int_slope = slope + 1.e0												#exponent of IMF integral		
	m_lower = m_lower**(int_slope)
	norm = m_upper**(int_slope) - m_lower									#PDF normalisation between m_upper and m_lower
	prob = np.random(Nstar)													#random number used to assign mass

	masses = (prob * norm - m_lower) ** 1.e0 / int_slope 					#convert random numbers to stellar masses	

	return masses, seed


def sample_linear_SFH(t_max = 400, f_sfr = 0, Nstar = 1000):
	
	grad = (f_sfr - 1.e0) / t_max										#SFR gradient over time
	norm = 1.e0 / (t_max + 0.5e0 * grad * t_max * t_max)				#PDF normalisation between t=0 and t_max

	prob = np.random(Nstar)												#random number for assigning formation time
																		#convert to formation time
	t_form = -1.e0 * norm + np.sqrt( norm * norm + 2.e0 * norm * grad * prob ) / (norm * grad)
	age = t_max - t_form

	return t_form, age, seed



def read_photometry(filename, BPs = ['f475w','f814w']):


	cols = [2,3]
	names = ['xpos', 'ypos']
	for ii in range(len(BPs)):
		iistep = 11*(ii+1) + ii 
		cols.extend([iistep, iistep + 4, iistep + 8, iistep + 9, iistep + 10, iistep + 11])
		names.extend(['{}_counts'.format(BPs[ii]),'{}_mag'.format(BPs[ii]),'{}_SN'.format(BPs[ii])
				,'{}_sharp'.format(BPs[ii]),'{}_round'.format(BPs[ii]),'{}_crowd'.format(BPs[ii])])


	phot = np.loadtxt(filename, usecols = cols)
	phot = Table(phot, names = names)

	print(phot)

	return phot


def read_ASTs(filename, BPs = ['f475w','f814w']):

	cols =  [10 + 10*(len(BPs)) + 2*(len(BPs) - 2),10 + 10*(len(BPs)) + 2*(len(BPs) - 2) + 1]
						
	names = ['xpos','ypos']

	for ii in range(len(BPs)):
		cols.extend([5*(ii + 1) + ii])
		names.extend(['{}_mag_in'.format(BPs[ii])])
		iistep = 38 + 12 * (ii + len(BPs) - 2) + (ii + 1)
		cols.extend([iistep, iistep + 4, iistep + 8, iistep + 9, iistep + 10, iistep + 11])
		names.extend(['{}_counts'.format(BPs[ii]),'{}_mag'.format(BPs[ii]),'{}_SN'.format(BPs[ii])
				,'{}_sharp'.format(BPs[ii]),'{}_round'.format(BPs[ii]),'{}_crowd'.format(BPs[ii])])


	phot = np.loadtxt(filename, usecols = cols)
	phot = Table(phot, names = names)

	return phot


def quality_cuts(phot, BPs = ['f475w','f814w']):

	goodstars = []
	for ii in range(len(BPs)):
		BP = BPs[ii]
		goodstars.extend(np.where(table['{}_mag'.format(BP)] < 80)[0])

	for ii in range(len(BPs) - 1):
		for jj in range(ii + 1, len(BPs)):
			BP1 = BPs[ii]
			BP2 = BPs[jj]
			goodstars.extend(np.where(
							(table['{}_round'.format(BP1)]**2.e0 + table['{}_round'.format(BP2)]**2.e0)
								< 1.4**2.e0)[0])
			goodstars.extend(np.where(
							(table['{}_round'.format(BP1)]**2.e0 + table['{}_round'.format(BP2)]**2.e0)
								< 0.05**2.e0)[0])
			goodstars.extend(np.where(
							(table['{}_round'.format(BP1)]**2.e0 + table['{}_round'.format(BP2)]**2.e0)
								< 0.6**2.e0)[0])
	goodstars = np.unique(goodstars)
	phot['phot_cut'] = np.zeros(len(phot))
	phot['phot_cut'][goodstars] = 1
	phot = phot[goodstars]
	return phot


def medain_photometric_uncertainty(phot, BPs = ['f475w']):

	mag_bins = np.arange(20,27.5,0.5)

	phot = phot[phot['phot_cut'] == 1]

	phot_uncert = np.zeros([len(mag_bins), 2 * len(BPs)])

	names = []

	for ii in range(len(BPs)):
		BP = BPs[ii]
		mag_in = phot['{}_mag_in'.format(BP)]
		mag_out = phot['{}_mag'.format(BP)]
		names.extend(['{}_med_in'.format(BP),'{}_sigma'.format(BP)])

		for jj in range(len(mag_bins) - 1):
			mag_low = mag_bins[jj]
			mag_high = mag_bins[jj + 1]
			inbin = np.where((mag_in > mag_low) * (mag_in < mag_high))[0]
			mag_diff = (mag_out - mag_in)[mag_diff]
			phot_uncert[2*ii,jj] = np.median(mag_in)
			phot_uncert[2*ii+1,jj] = median_absolute_deviation(mag_diff)


	table = Table(phot_uncert, names = names)


	return table



def create_stellar_atmosphere_grid(BPs = ['acs,wfc1,f475w'], ZPs = [26.168e0], ebv = 0.01, Z = 0.01):
	import pysynphot as S
	BPs = [S.ObsBandpass(BP) for BP in BPs]
	spectra = glob.glob('./data/grp/hst/cdbs/grid/ck04models/ckm{Z}/ckm{Z}*.fits'.format(Z = int(Z*1000)))
	logg_range = np.arange(0,5.5,0.5)

	filter_mag_grids = {}

	for BP in range(len(BPs)):
		magnitudes = np.zeros([len(logg_range)*len(spectra),3])
		for ii in range(len(spectra)):
			spec_file = spectra[ii]
			Teff = file.split('_')[1].split('.')[0]
			for logg in logg_range:
				gg = 'g{0:02d}'.format(10*logg)
				base_spec = S.FileSpectrum(spec_file,fluxname = gg)
				base_spec = base_spec * (6.957e8 / 3.0857e19)**2.e0
				if sum(base_spec.flux) != 0:
					spec = base_spec * S.Extinction(ebv,'gal3')
					obs = S.Observation(spec,BPs[BP])
					count = obs_475.countrate()
					mag = -2.5e0 * np.log10(count) + ZPs[BP]
				else:
					mag = np.nan

				magnitudes[0] = logg
				magnitdues[1] = Teff
				magnitudes[2] = mag
			
				print('done {}'.format(gg))
			print('done {}'.format(Teff))

		filter_mag_grids[BPs[BP]] = magnitues

	f = open('model_atmosphere_ebv{ebv:.3f}_Z{Z:.3f}.dat'.format(ebf=ebv,Z=Z),'w')
	f.write(str(filter_mags))
	f.close()


def median_absolute_deviation(array):
	MAD = np.median(np.abs( array - np.median(array))) * 1.365
	return MAD


if __name__ == '__main__':
	
	# read_photometry('/home/awatts/Adam_PhD/MastersWork/data/DDO006.dat')
	# read_ASTs('/home/awatts/Adam_PhD/MastersWork/data/DDO006f6.dat')
	sample_IMF_ratio()
