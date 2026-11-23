"""Solar elevation at a site, used to take the control quantile of the default word at night.

In the day the Sun is common to every input of the array, its fluctuations add coherently in the summed fine
powers, and the scores of the control rise. CHIME's 21 cm analyses use only the data with the Sun below the
horizon (CHIME Collaboration 2023, ApJ 947, 16, Sec. III.2.7), and the night option of the default word follows
them.

The position of the Sun is the low-precision formula of the Astronomical Almanac, as the NOAA solar calculator
uses it: about 0.01 deg over 1950-2050. Refraction is left out, so the Sun is below the horizon when its geometric
elevation is below zero.
"""
from __future__ import annotations

import numpy as np

UNIX_EPOCH_JD = 2440587.5
J2000_JD = 2451545.0


def elevation_deg(unix_s, latitude_deg: float, longitude_deg_east: float) -> np.ndarray:
    """Return the geometric elevation of the Sun's centre in degrees at Unix times unix_s (seconds, UTC)."""
    n = np.asarray(unix_s, dtype=np.float64) / 86400.0 + UNIX_EPOCH_JD - J2000_JD
    mean_longitude = np.radians((280.460 + 0.9856474 * n) % 360.0)
    anomaly = np.radians((357.528 + 0.9856003 * n) % 360.0)
    ecliptic_longitude = mean_longitude + np.radians(1.915) * np.sin(anomaly) + np.radians(0.020) * np.sin(2 * anomaly)
    obliquity = np.radians(23.439 - 4e-7 * n)
    right_ascension = np.arctan2(np.cos(obliquity) * np.sin(ecliptic_longitude), np.cos(ecliptic_longitude))
    declination = np.arcsin(np.sin(obliquity) * np.sin(ecliptic_longitude))
    sidereal = np.radians(((18.697374558 + 24.06570982441908 * n) % 24.0) * 15.0)
    hour_angle = sidereal + np.radians(longitude_deg_east) - right_ascension
    latitude = np.radians(latitude_deg)
    return np.degrees(np.arcsin(np.sin(latitude) * np.sin(declination)
                                + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)))


def is_night(unix_s, latitude_deg: float, longitude_deg_east: float) -> np.ndarray:
    """Return True where the Sun is below the horizon at the site."""
    return elevation_deg(unix_s, latitude_deg, longitude_deg_east) < 0.0
