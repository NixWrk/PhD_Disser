"""Signal-level checks against known synthetic components and event times."""
import unittest

import numpy as np

from scorpion_qc import bei_filter, ecg_filter, qrs_candidates


class SignalChecks(unittest.TestCase):
    def test_notch_preserves_passband_and_rejects_known_mains(self):
        for fs in (400, 1000):
            t = np.arange(30 * fs) / fs
            x = 2 + np.sin(2*np.pi*3*t)
            for hz in (50, 100, 150):
                x += .8*np.sin(2*np.pi*hz*t)
            notch, _ = ecg_filter(x, fs)
            interior = slice(5*fs, -5*fs)
            amplitude3 = abs(2*np.mean(notch[interior]*np.exp(-2j*np.pi*3*t[interior])))
            self.assertAlmostEqual(amplitude3, 1, delta=.005)
            for hz in (50, 100, 150):
                amplitude = abs(2*np.mean(notch[interior]*np.exp(-2j*np.pi*hz*t[interior])))
                self.assertLess(amplitude, .008)

    def test_bei_dc_and_known_slow_pulse_components(self):
        fs = 1000; t = np.arange(40*fs)/fs
        slow_true = 70 + .5*np.sin(2*np.pi*.15*t)
        pulse_true = .02*np.sin(2*np.pi*2*t)
        raw = slow_true + pulse_true
        slow, residual, pulse, other, _ = bei_filter(raw, fs)
        interior = slice(8*fs, -8*fs)
        self.assertLess(np.max(np.abs(raw - (slow + pulse + other))), 1e-11)
        self.assertLess(np.sqrt(np.mean((slow[interior]-slow_true[interior])**2)), .003)
        self.assertLess(np.sqrt(np.mean((pulse[interior]-pulse_true[interior])**2)), .001)
        self.assertLess(np.max(np.abs(raw-(slow+residual))), 1e-11)

    def test_qrs_candidates_against_known_events_with_noise_and_mains(self):
        rng = np.random.default_rng(6107)
        for fs in (400, 1000):
            t = np.arange(30*fs)/fs
            times = np.arange(2, 28, .8)
            x = .02*rng.standard_normal(len(t))
            for hz in (50, 100, 150):
                x += .7*np.sin(2*np.pi*hz*t)
            for i, beat in enumerate(times):
                x += (1 if i%3 else -1)*np.exp(-.5*((t-beat)/.015)**2)
                x += .15*np.exp(-.5*((t-beat-.22)/.05)**2)
            _, filtered = ecg_filter(x, fs)
            detected = qrs_candidates(filtered, fs)/fs
            errors = np.abs(times[:,None]-detected[None,:])
            self.assertGreaterEqual(np.mean(errors.min(axis=1)<.03), .95)
            self.assertGreaterEqual(np.mean(errors.min(axis=0)<.03), .95)


if __name__ == '__main__': unittest.main()
