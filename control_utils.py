"""Shared measured-goal preprocessing; no direction or wheel planning."""
import numpy as np
def standoff(height):
 return np.interp(height,[.1,.5,1.1,1.35],[.8,.65,.65,.205])
