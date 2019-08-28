






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


