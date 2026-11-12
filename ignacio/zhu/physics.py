"""What the measured numbers mean physically: the inspiral time between precursor and main emission, the size the variability allows,
and the energetics and shattering-flare budget where a redshift is known (the last two from grbtool.physics, unchanged).

Inspiral time. Two compact objects reach a gravitational-wave frequency f at a time

    tau = (5/256) (G M_c / c^3)^(-5/3) (pi f)^(-8/3)

before they merge (leading order, the same formula as the course notes: 2.2 s at 100 Hz for two 1.4 solar-mass stars). In the resonant
shattering picture (Tsang et al. 2012) the crust breaks when f reaches the frequency of the crust-core interface mode, so the precursor
starts tau before the merger. The main emission can't start before the merger, so the time from the start of the precursor to the start
of the main emission, in the burst's rest frame, is at least tau. That puts a lower limit on the mode frequency, for an assumed chirp
mass. Without a redshift the observed delay is used, which gives the lowest (safest) limit, since any redshift shortens the rest-frame
delay and raises the limit.

Size. Variability on a time dt (in the rest frame) means the region that changes is at most about c dt across, or 2 Gamma^2 c dt if it
moves towards us with Lorentz factor Gamma.
"""
import numpy as np
import pandas as pd
from astropy import constants as const

G_MSUN_OVER_C3 = (const.G * const.M_sun / const.c**3).to_value("s")        # 4.93e-6 s
C_KM_S = const.c.to_value("km/s")


def chirp_mass(m1, m2):
    """Chirp mass in solar masses."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


BNS = chirp_mass(1.4, 1.4)          # 1.22
NSBH = chirp_mass(1.4, 5.0)         # 2.22


def inspiral_time(f_gw, mc=BNS):
    """Time left before merger, in s, at gravitational-wave frequency f_gw (Hz), for chirp mass mc (solar masses)."""
    return 5.0 / 256.0 * (G_MSUN_OVER_C3 * mc) ** (-5.0 / 3.0) * (np.pi * np.asarray(f_gw, float)) ** (-8.0 / 3.0)


def frequency_at(tau, mc=BNS):
    """The gravitational-wave frequency (Hz) a time tau (s) before merger: the inverse of inspiral_time."""
    return (5.0 / 256.0 * (G_MSUN_OVER_C3 * mc) ** (-5.0 / 3.0) / np.asarray(tau, float)) ** 0.375 / np.pi


def delay_limits(ours, z=None):
    """For each burst with a precursor: the delay from the precursor's start to the main emission's start (observed, and in the rest frame
    if z is given), and the lowest interface-mode frequency it allows for a neutron-star pair and for a neutron star with a 5 solar-mass
    black hole. ours needs pe_start and me_start; z is a Series of redshifts (missing ones use the observed delay)."""
    out = pd.DataFrame(index=ours.index)
    out["delay_obs"] = ours.me_start - ours.pe_start
    redshift = pd.Series(np.nan, index=ours.index) if z is None else z.reindex(ours.index)
    out["z"] = redshift
    out["delay_rest"] = out.delay_obs / (1 + redshift.fillna(0.0))
    out["f_min_bns"] = frequency_at(out.delay_rest, BNS)
    out["f_min_nsbh"] = frequency_at(out.delay_rest, NSBH)
    return out


def light_size_km(dt, z=None, gamma=1.0):
    """The largest size (km) a region varying on observed time dt (s) can have: 2 Gamma^2 c dt / (1+z), or c dt / (1+z) for Gamma = 1."""
    factor = 1.0 if gamma == 1.0 else 2.0 * gamma**2
    return factor * C_KM_S * np.asarray(dt, float) / (1.0 + (0.0 if z is None else np.asarray(z, float)))
