"""Display filtering preserves the raw record and isolates known frequency effects."""
import unittest
import numpy as np
from experiment_signal_display import ecg_variants, spectrum, phase_intervals


class SignalDisplayChecks(unittest.TestCase):
    def test_filters_can_be_disabled_and_match_the_existing_default(self):
        from scorpion_qc import ecg_filter
        for fs in (400,1000):
            t=np.arange(24*fs)/fs;x=np.sin(2*np.pi*8*t)+.4*np.sin(2*np.pi*50*t)+.03*t
            untouched=x.copy();variants=ecg_variants(x,fs)
            np.testing.assert_array_equal(x,untouched)
            np.testing.assert_array_equal(variants['00'],x)
            np.testing.assert_allclose(variants['11'],ecg_filter(x,fs)[1],atol=1e-12)

    def test_notches_and_band_have_distinct_effects_in_synthetic_spectrum(self):
        fs=400;t=np.arange(32*fs)/fs
        x=sum(np.sin(2*np.pi*f*t) for f in (.125,8,50,100,150))
        variants=ecg_variants(x,fs);psd={}
        for key,y in variants.items():
            f,p=spectrum(y[4*fs:-4*fs],fs);psd[key]=p
        at=lambda key,hz:psd[key][np.argmin(abs(f-hz))]
        for hz in (50,100,150):self.assertLess(at('10',hz),at('00',hz)*.001)
        self.assertGreater(at('10',8),at('00',8)*.95)
        self.assertGreater(at('10',.125),at('00',.125)*.9)
        self.assertLess(at('01',.125),at('00',.125)*.01)
        self.assertGreater(at('01',8),at('00',8)*.95)
        self.assertLess(at('01',150),at('00',150)*.001)

    def test_welch_power_has_physical_density_units(self):
        fs=1000;t=np.arange(24*fs)/fs;f,p=spectrum(np.sqrt(2)*np.sin(2*np.pi*8*t),fs)
        self.assertEqual(f[-1],fs/2)
        self.assertAlmostEqual(np.trapezoid(p,f),1.,places=5)

    def test_phase_edges_use_the_record_timebase_without_acceptance(self):
        candidate={'boundaries_samples':[8800,16800,22800],'author_accepted':False}
        intervals=phase_intervals(candidate,400,80)
        self.assertEqual([p['end'] for p in intervals],[22,42,57,80])
        self.assertFalse(candidate['author_accepted'])
        self.assertEqual(len(phase_intervals({'single_mode':'calm_breathing'},1000,23)),1)
        self.assertEqual(phase_intervals({},1000,80),[])
        with self.assertRaises(ValueError):phase_intervals({'boundaries_samples':[3,2,1]},1,10)


if __name__=='__main__':unittest.main()
