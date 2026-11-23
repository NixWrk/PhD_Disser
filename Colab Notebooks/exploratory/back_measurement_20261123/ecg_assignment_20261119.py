"""ECG-only channel-assignment audit for the 25 exports dated 2026-11-19.

This is a separate, exploratory analysis, not the canonical archive parser.
The canonical parser loads the export; only COUNT, EMG1mV and EMG2mV
are passed into this analysis. No impedance samples are used.
H1: channel 1 stays on the right posterior assembly; channel 2 switches
between arm TTRKG and left posterior. H2 is the reversed assignment.
EMG1/2 and BEI1/2 correspondence is an author-supplied wiring fact.
The result tests relative channel continuity, not absolute anatomy.

Run with --archive PATH. By default only the aggregate CSV beside this script
is written. --debug-dir PATH explicitly enables local diagnostic plots. Signal
arrays are never serialized. --reference-root ROOT locates the optional prior
records under ROOT/Experiments; its default is this script's repository root.
Missing historical references are marked, not silently treated as comparisons.
CSV row_type distinguishes records, matched seated pairs, and size pairs.
The ECG detector is only for this audit, not the pulse-dZ pipeline.

Method: 50-Hz notch (Q=30), zero-phase order-4 0.7-35-Hz Butterworth;
QRS-energy detection at 8-25 Hz; common within-file anchor; median beats
from -250 to +500 ms, after -250..-170-ms baseline subtraction. Retain
beats with QRS correlation >=0.6 and amplitude 0.35..2.5 times the median.
The QRST comparison uses -120..+400 ms, QRS uses -100..+120 ms, with
translation <=30 ms, no time stretching, and no heart-rate matching.
Morphology is usable if >=60% beats survive, median QRS correlation >=0.6,
at least four of six time blocks contribute, and minimum block correlation
to the overall template is >=0.7. The thresholds are descriptive QC rules.
Block min/max and win fractions describe six chronological blocks; they
are NOT confidence intervals, independent repeats, or calibrated probabilities.
Dominant-peak FWHM is a filtered shape feature, NOT clinical QRS duration.

Public evidence codes: ECG_SWITCH (same-size seated contrasts), ECG_SIZE,
ECG_LYING100 (author-confirmed bilateral lying control), ECG_POSTURE,
ECG_QC, ECG_NOISE_CONTROL, ECG_PRIOR_U2, ECG_PRIOR_POLARITY_LIMIT.
Assignment codes distinguish direct switch support, posture continuity,
ECG_60_CH2_ONLY, and ECG_60_PROTOCOL_TRANSFER. The last is explicitly not
an independent ECG assignment. H1/H2 assume no unreported channel-specific
polarity/gain change between paired exports. Arbitrarily allowing both
would reduce identifiability, particularly for U1. Lying does not specify
which side of the body supports the participant.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import zipfile
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d
from analyze_recordings_20261119 import iter_exports, identify, SIZES


VERSION = "ecg-assignment-20261119-v4"

# Saved protocol assigns these physical channels independently of this audit.
# Historical filenames are retained exactly, including superseded 1x4 labels.
REFERENCES = (
    ("U1", "ECG22_U1_ISOLATED_2X2", "Experiments/scorpion_20261022/raw/Scorp_22_10_26/Ник/только ттркг 2х2.txt", 1,
     "isolated_probe_analysis_and_author_confirmed_session_continuity"),
    ("U1", "ECG22_U1_1X1", "Experiments/scorpion_20261022/raw/Scorp_22_10_26/Ник/оба канала 60 1х1.txt", 1,
     "session_continuity_from_isolated_probe_not_direct_recall"),
    ("U2", "ECG05_U2_1X1", "Experiments/scorpion_v2_georg/raw/Георг_05_10_26/120 горизонт 1х4.txt", 2,
     "physical_channel_author_confirmed_20261009_actual_1x1_confirmed_20261014"),
)


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
    return np.sum(a*b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1), 1e-20)


def similarity(a, b, fs, window=(-.12, .40), absolute=False):
    """Pearson correlation in fixed seconds, with at most 30 ms translation.

    Primary comparisons preserve polarity. Absolute correlation is explicitly
    a sensitivity analysis allowing a lead reversal, never primary evidence.
    There is no RR normalization/time stretching and no raw-record correlation.
    """
    t = np.arange(len(a))/fs - .25
    ix = np.flatnonzero((t >= window[0]) & (t <= window[1]))
    shifts = np.arange(-round(.03*fs), round(.03*fs)+1, max(1, round(.002*fs)))
    scores = np.array([corr(a[ix], b[ix+s]) for s in shifts])
    k = np.argmax(abs(scores) if absolute else scores)
    return float(abs(scores[k]) if absolute else scores[k]), float(shifts[k]/fs*1000)


def detect(x, fs):
    """Independent QRS-energy detections for each lead; no clinical labels."""
    q = signal.sosfiltfilt(signal.butter(3, [8, 25], fs=fs, btype="bandpass", output="sos"), x)
    energy = uniform_filter1d(q*q, size=round(.08*fs))
    med = np.median(energy)
    mad = np.median(abs(energy-med))
    peaks, _ = signal.find_peaks(energy, distance=round(.40*fs), prominence=max(3*mad, .05*np.percentile(energy, 95)), height=med+2*mad)
    peaks = peaks[(peaks > fs) & (peaks < len(x)-fs)]
    # Align to largest QRS excursion, without changing its polarity.
    aligned = np.array([p-round(.065*fs)+np.argmax(abs(q[p-round(.065*fs):p+round(.065*fs)+1])) for p in peaks])
    # Remove small energy maxima within noisy intervals, using the recording's
    # typical detected QRS amplitude. This cannot rescue the corrupted 60-mm leads.
    amplitude = abs(q[aligned])
    return aligned[amplitude >= .40*np.median(amplitude)], q


def beat_stack(x, peaks, fs):
    offsets = np.arange(-round(.25*fs), round(.50*fs)+1)
    stack = x[peaks[:, None]+offsets]
    return stack-np.median(stack[:, :round(.08*fs)], axis=1)[:, None]


def process(raw, fs):
    # Powerline notch followed by a zero-phase 0.7-35 Hz ECG bandpass.
    b, a = signal.iirnotch(50, 30, fs)
    x = signal.filtfilt(b, a, raw, axis=0)
    x = signal.sosfiltfilt(signal.butter(4, [.7, 35], fs=fs, btype="bandpass", output="sos"), x, axis=0)
    det = [detect(x[:, j], fs)[0] for j in range(2)]
    independent = []
    for j in range(2):
        st = beat_stack(x[:, j], det[j], fs)
        independent.append(float(np.median(corr(st, np.median(st, axis=0)))))
    reference = int(np.argmax(independent))
    peaks = det[reference]
    stacks = [beat_stack(x[:, j], peaks, fs) for j in range(2)]
    templates, blocks, quality = [], [], []
    for j, st in enumerate(stacks):
        seed = np.median(st, axis=0)
        c = corr(st, seed)
        qi = slice(round(.15*fs),round(.37*fs))
        qc = corr(st[:,qi],seed[qi])
        amplitude = np.ptp(st, axis=1)
        amed = np.median(amplitude)
        good = (qc >= .60) & (amplitude > .35*amed) & (amplitude < 2.5*amed)
        # A poor lead remains explicitly poor; no silent selection of a few beats.
        template = np.median(st[good], axis=0) if good.sum() >= 5 else seed
        templates.append(template)
        bt = []
        for k in range(6):
            mask = good & (peaks >= k*len(x)/6) & (peaks < (k+1)*len(x)/6)
            if mask.sum() >= 3:
                bt.append(np.median(st[mask], axis=0))
        blocks.append(bt)
        qrs = template[round(.13*fs):round(.37*fs)]
        qmax = np.argmax(abs(qrs))
        lo,hi=qmax,qmax
        while lo>0 and abs(qrs[lo-1])>=.5*abs(qrs[qmax]):
            lo-=1
        while hi<len(qrs)-1 and abs(qrs[hi+1])>=.5*abs(qrs[qmax]):
            hi+=1
        twave=template[round(.40*fs):round(.65*fs)]
        f, psd = signal.welch(raw[:, j], fs=fs, nperseg=min(len(raw), round(4*fs)))
        line = np.sum(psd[(f >= 49) & (f <= 51)]) / np.maximum(np.sum(psd[(f >= .7) & (f <= 100)]), 1e-20)
        residual = st[good]-template if good.any() else st-template
        snr = np.sqrt(np.mean(template**2))/max(np.sqrt(np.median(np.mean(residual**2, axis=1))), 1e-15)
        split_min=float(min([similarity(template, v, fs)[0] for v in bt], default=np.nan))
        usable=bool(good.mean()>=.60 and np.median(qc)>=.60 and split_min>=.70 and len(bt)>=4)
        quality.append(dict(channel=j+1, detected_beats=len(peaks), retained_beats=int(good.sum()),
            morphology_usable=usable,
            retained_fraction=float(good.mean()), beat_corr_median=float(np.median(c)),
            beat_corr_p10=float(np.percentile(c, 10)), qrs_beat_corr_median=float(np.median(qc)), independent_beats=len(det[j]),
            independent_hr_bpm=float(60*fs/np.median(np.diff(det[j]))),
            hr_bpm=float(60*fs/np.median(np.diff(peaks))),
            rr_cv=float(np.std(np.diff(peaks))/np.mean(np.diff(peaks))),
            template_ptp_mV=float(np.ptp(template)), qrs_max_mV=float(qrs.max()),
            qrs_min_mV=float(qrs.min()), qrs_polarity_balance=float((qrs.max()+qrs.min())/(qrs.max()-qrs.min())),
            dominant_qrs_peak_fwhm_ms=(hi-lo+1)/fs*1000, t_to_qrs_signed=float(twave[np.argmax(abs(twave))]/abs(qrs[qmax])),
            raw_median_mV=float(np.median(raw[:, j])), raw_line_power_fraction=float(line),
            template_to_residual_rms=float(snr), reference_ecg_channel=reference+1,
            split_corr_min=split_min,
            flags=";".join(s for test, s in [(good.mean()<.70, "LOW_RETAINED"), (np.median(c)<.90, "LOW_MORPHOLOGY_REPEATABILITY"),
                (line>.50, "POWERLINE_DOMINANT_RAW"), (abs(np.median(raw[:, j]))>100, "LARGE_DC_OFFSET"),
                (np.std(np.diff(peaks))/np.mean(np.diff(peaks))>.15, "RR_IRREGULAR_OR_DETECTION")] if test)))
    near = [np.mean(np.min(abs(p[:, None]-other[None, :]), axis=1) <= .06*fs) for p, other in [(det[0],det[1]), (det[1],det[0])]]
    return dict(x=x, templates=np.array(templates), blocks=blocks, peaks=peaks,
                quality=quality, independent_qrs_match_fraction=float(min(near)), stacks=stacks)


def add_comparisons(records, fs, rows):
    bykey = {(r["subject"], r["posture"], r["configuration"], r["size_mm"]): r for r in records}
    for subject in ("U1", "U2"):
        for size in SIZES:
            a = bykey[(subject, "seated", "back_arm", size)]
            b = bykey[(subject, "seated", "bilateral", size)]
            for absolute in (False, True):
                for window, wn in [((-.12,.40), "QRST"), ((-.10,.12), "QRS")]:
                    scores = [similarity(a["templates"][j], b["templates"][j], fs, window, absolute)[0] for j in range(2)]
                    signed_at_abs=[]
                    for j in range(2):
                        _,lag=similarity(a["templates"][j],b["templates"][j],fs,window,True)
                        ix=np.flatnonzero((np.arange(a["templates"].shape[1])/fs-.25>=window[0]) &
                                          (np.arange(a["templates"].shape[1])/fs-.25<=window[1]))
                        signed_at_abs.append(float(corr(a["templates"][j,ix],b["templates"][j,ix+round(lag*fs/1000)])))
                    differences = []
                    for k in range(min(len(a["blocks"][0]),len(a["blocks"][1]))):
                        for l in range(min(len(b["blocks"][0]),len(b["blocks"][1]))):
                            differences.append(similarity(a["blocks"][0][k], b["blocks"][0][l], fs, window, absolute)[0]-similarity(a["blocks"][1][k], b["blocks"][1][l], fs, window, absolute)[0])
                    rows.append(dict(row_type="matched_seated_pair", evidence_code="ECG_SWITCH", subject=subject,
                        size_mm=size, record_id=a["record_id"], comparator_id=b["record_id"], window=wn, allow_polarity_reversal=absolute,
                        rho_ch1=scores[0], rho_ch2=scores[1], h1_minus_h2=scores[0]-scores[1],
                        rho_ch1_signed_at_abs_lag=signed_at_abs[0],rho_ch2_signed_at_abs_lag=signed_at_abs[1],
                        amplitude_ratio_ch1=float(np.ptp(a["templates"][0])/np.ptp(b["templates"][0])),
                        amplitude_ratio_ch2=float(np.ptp(a["templates"][1])/np.ptp(b["templates"][1])),
                        block_delta_min=min(differences,default=np.nan), block_delta_max=max(differences,default=np.nan), block_h1_win_fraction=float(np.mean(np.array(differences)>0)) if differences else np.nan,
                        block_comparisons=len(differences), status="ECG_60_UNUSABLE" if size==60 else "relative_continuity_not_absolute_position"))
        for posture in ("seated", "lying"):
            for config in ("back_arm", "bilateral"):
                group = [r for r in records if r["subject"]==subject and r["posture"]==posture and r["configuration"]==config]
                for i,a in enumerate(group):
                    for b in group[i+1:]:
                        for j in range(2):
                            rho, lag = similarity(a["templates"][j], b["templates"][j], fs)
                            rows.append(dict(row_type="cross_size", evidence_code="ECG_SIZE", subject=subject,
                                posture=posture, configuration=config, record_id=a["record_id"], comparator_id=b["record_id"],
                                channel=j+1, size_mm=a["size_mm"], comparator_size_mm=b["size_mm"], rho=rho, lag_ms=lag,
                                comparison_usable=bool(a["quality"][j]["morphology_usable"] and b["quality"][j]["morphology_usable"])))

    # Separate confirmation in the one author-confirmed bilateral lying export.
    a=bykey[("U2","lying","back_arm",100)]
    b=bykey[("U2","lying","bilateral",100)]
    for j in range(2):
        rows.append(dict(row_type="lying_switch",evidence_code="ECG_LYING100",subject="U2",channel=j+1,size_mm=100,
            record_id=a["record_id"],comparator_id=b["record_id"],rho=similarity(a["templates"][j],b["templates"][j],fs)[0],
            amplitude_ratio=float(np.ptp(a["templates"][j])/np.ptp(b["templates"][j]))))

    for r in records:
        if r["size_mm"]==60:
            code="ECG_60_CH2_ONLY" if r["configuration"]=="back_arm" else "ECG_60_PROTOCOL_TRANSFER"
        elif r["posture"]=="lying":
            code="ECG_H1_LYING100_SUPPORTED" if r["subject"]=="U2" and r["size_mm"]==100 else "ECG_H1_POSTURE_CONTINUITY"
        else:
            code="ECG_H1_SWITCH_SUPPORTED"
        rows.append(dict(row_type="assignment",evidence_code=code,
            subject=r["subject"],posture=r["posture"],configuration=r["configuration"],size_mm=r["size_mm"],record_id=r["record_id"],
            preferred_hypothesis="H1",h1_bei1="right_posterior",h1_bei2="arm_ttrkg" if r["configuration"]=="back_arm" else "left_posterior",
            h2_bei2="right_posterior",h2_bei1="arm_ttrkg" if r["configuration"]=="back_arm" else "left_posterior",
            status="protocol_transfer_no_independent_ecg_assignment" if code=="ECG_60_PROTOCOL_TRANSFER" else "relative_assignment_supported_protocol_conditional_not_absolute_position",
            flags="NO_USABLE_ECG_BILATERAL_60" if code=="ECG_60_PROTOCOL_TRANSFER" else ""))

    # A fixed anatomical rule based only on generic QRS appearance is not used:
    # within-record lead similarity explicitly measures this limitation.
    for r in records:
        for absolute in (False,True):
            rho,lag=similarity(r["templates"][0],r["templates"][1],fs,absolute=absolute)
            rows.append(dict(row_type="within_record_leads",evidence_code="ECG_NOT_ANATOMICAL_CLASSIFIER",
                subject=r["subject"],record_id=r["record_id"],size_mm=r["size_mm"],posture=r["posture"],configuration=r["configuration"],
                rho=rho,lag_ms=lag,allow_polarity_reversal=absolute,
                comparison_usable=bool(all(q["morphology_usable"] for q in r["quality"]))))

    # Control for size and posture: the arm configuration stays 1x1/100 mm;
    # the right posterior assembly is allowed to change with size.
    for subject in ("U1","U2"):
        for size in SIZES:
            a=bykey[(subject,"seated","back_arm",size)]
            b=bykey[(subject,"lying","back_arm",size)]
            for j in range(2):
                rows.append(dict(row_type="posture",evidence_code="ECG_POSTURE",subject=subject,size_mm=size,channel=j+1,
                    record_id=a["record_id"],comparator_id=b["record_id"],rho=similarity(a["templates"][j],b["templates"][j],fs)[0],
                    status="unusable_channel" if not (a["quality"][j]["morphology_usable"] and b["quality"][j]["morphology_usable"]) else "comparison"))


def add_references(records, fs, rows, root):
    for subject,rid,relative,ch,provenance in REFERENCES:
        path=root/relative
        if not path.exists():
            rows.append(dict(row_type="historical_reference",evidence_code="ECG_PRIOR",subject=subject,
                             comparator_id=rid,status="missing_optional_reference"))
            continue
        data=path.read_bytes()
        raw=np.loadtxt(io.BytesIO(data.replace(b",",b".")),skiprows=1,usecols=(0,ch))
        if np.any(np.diff(raw[:,0])%256!=1):
            raise ValueError(f"Noncontinuous historical ECG: {rid}")
        # Duplicate only the selected ECG for this single-lead template builder.
        result=process(np.column_stack([raw[:,1],raw[:,1]]),1000)
        template=result["templates"][0]
        if fs!=1000:
            template=np.interp(np.arange(round(.75*fs)+1)/fs,np.arange(len(template))/1000,template)
        for r in records:
            if r["subject"]!=subject:
                continue
            for j in range(2):
                for absolute in (False,True):
                    rho,lag=similarity(template,r["templates"][j],fs,absolute=absolute)
                    rows.append(dict(row_type="historical_reference",evidence_code="ECG_PRIOR_POLARITY_LIMIT" if subject=="U1" else "ECG_PRIOR_U2",
                        subject=subject,record_id=r["record_id"],comparator_id=rid,channel=j+1,size_mm=r["size_mm"],
                        posture=r["posture"],configuration=r["configuration"],rho=rho,lag_ms=lag,allow_polarity_reversal=absolute,
                        source_sha256=hashlib.sha256(data).hexdigest(),reference_ecg_channel=ch,
                        reference_provenance=provenance,reference_retained_beats=result["quality"][0]["retained_beats"],
                        reference_median_beat_corr=result["quality"][0]["beat_corr_median"],
                        status="cross_session_diagnostic_not_absolute_position_reference"))


def add_noise_controls(records,fs,rows):
    """Uniform event jitter removes beat locking while preserving recording noise.

    This is a diagnostic coherence ratio, not a p-value: QRS was detected in
    these same data. The opposite lead's independently detected timing is used
    to avoid creating a noise peak by aligning a lead to its own extrema.
    """
    rng=np.random.default_rng(20261119)
    for r in records:
        for j in range(2):
            other,_=detect(r["x"][:,1-j],fs)
            st=beat_stack(r["x"][:,j],other,fs)
            med=np.median(st,axis=0)
            qi=slice(round(.15*fs),round(.37*fs))
            observed=float(np.ptp(med[qi]))
            null=[]
            for _ in range(100):
                jitter=rng.integers(-round(.35*fs),round(.35*fs)+1,len(other))
                null.append(np.ptp(np.median(beat_stack(r["x"][:,j],other+jitter,fs),axis=0)[qi]))
            rows.append(dict(row_type="beat_locking",evidence_code="ECG_NOISE_CONTROL",subject=r["subject"],record_id=r["record_id"],
                channel=j+1,size_mm=r["size_mm"],posture=r["posture"],configuration=r["configuration"],
                cross_lead_locked_qrs_ptp_mV=observed,jitter_qrs_ptp_p95_mV=float(np.percentile(null,95)),
                locking_to_jitter_p95_ratio=observed/np.percentile(null,95),jitter_replicates=len(null),
                reference_ecg_channel=2-j,reference_morphology_usable=r["quality"][1-j]["morphology_usable"],
                status="descriptive_noise_control_not_statistical_significance"))


def plot_records(records, fs, debug):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for subject in ("U1", "U2"):
        selected = [r for r in records if r["subject"]==subject]
        fig, axes = plt.subplots(len(selected), 3, figsize=(15, 2.2*len(selected)))
        for ax, r in zip(axes, selected):
            x = r["x"]
            t = np.arange(len(x))/fs
            for j in range(2):
                w = (t>=10)&(t<=14)
                ax[j].plot(t[w], x[w,j], lw=.7)
                p=r["peaks"][(r["peaks"]>=10*fs)&(r["peaks"]<=14*fs)]
                ax[j].plot(p/fs,x[p,j],"r.")
                ax[j].set_title(f'{r["record_id"]} EMG{j+1} good={r["quality"][j]["retained_fraction"]:.2f}')
                temp=r["templates"][j]
                ax[2].plot(np.arange(len(temp))/fs-.25, temp/max(np.ptp(temp),1e-12), label=f'EMG{j+1}')
            ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(debug/f"ecg_{subject}.png",dpi=110)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--fs",type=float,default=1000,help="Nominal clock from the matched BIN intervals and file-time check")
    p.add_argument("--csv",type=Path,default=Path(__file__).with_suffix(".csv"))
    p.add_argument("--debug-dir",type=Path,help="Explicitly enable local plots in an ignored directory; no signal serialization")
    p.add_argument("--reference-root",type=Path,default=Path(__file__).resolve().parents[3])
    args=p.parse_args()
    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True,exist_ok=True)
        # This directory is specifically authorized for ignored local diagnostics.
        (args.debug_dir/".gitignore").write_text("*\n",encoding="utf-8")
    rows,records=[],[]
    archive_sha=hashlib.sha256(args.archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(args.archive) as z:
        source={identify(i.filename):(i.filename,hashlib.sha256(z.read(i)).hexdigest())
                for i in z.infolist() if i.filename.lower().endswith(".txt")}
        for key, exported in iter_exports(args.archive):
            participant,posture,pair,size=key
            a=exported[:,:3]
            gaps=int(np.sum(np.diff(a[:,0])%256!=1))
            if gaps or not np.all(np.isfinite(a)):
                raise ValueError("Noncontinuous counter or nonfinite ECG")
            meta=dict(subject=participant,posture=posture,configuration=pair,size_mm=size)
            rid=f'ECG19_{participant}_{posture}_{pair}_{size:03d}'
            r=dict(**meta,record_id=rid,**process(a[:,1:3],args.fs))
            records.append(r)
            for q in r["quality"]:
                rows.append(dict(row_type="record",evidence_code="ECG_QC",**meta,record_id=rid,
                    participant=participant,source_posture=posture,pair=pair,
                    source_sha256=source[key][1],
                    archive_sha256=archive_sha,samples=len(a),duration_s=len(a)/args.fs,
                    counter_nonunit_steps=gaps,fs_hz=args.fs,fs_provenance="all_25_TXT_BIN_exact_match_time_check_999.751_Hz_author_20261123",
                    independent_qrs_match_fraction=r["independent_qrs_match_fraction"],**q))
            print(rid, [(q["retained_beats"],round(q["beat_corr_median"],3),round(q["template_ptp_mV"],3)) for q in r["quality"]],flush=True)
    add_comparisons(records,args.fs,rows)
    add_references(records,args.fs,rows,args.reference_root)
    add_noise_controls(records,args.fs,rows)
    df=pd.DataFrame(rows)
    df["algorithm_version"]=VERSION
    # Git stores these two code files with LF even if the working copy uses CRLF.
    script = Path(__file__)
    df["code_sha256"]=hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    df["archive_sha256"]=archive_sha
    parser_source = script.with_name("analyze_recordings_20261119.py").read_bytes().replace(b"\r\n", b"\n")
    df["parser_sha256"]=hashlib.sha256(parser_source).hexdigest()
    df["fs_hz"]=args.fs
    # Public output is aggregate-only: no filenames, paths, waveform samples,
    # template arrays, or per-beat timestamps. Provenance uses hashes and codes.
    if "source_member" in df.columns:
        raise ValueError("Private source names must not enter public CSV")
    df.to_csv(args.csv,index=False,encoding="utf-8-sig",float_format="%.9g",lineterminator="\n")
    if args.debug_dir is not None:
        plot_records(records,args.fs,args.debug_dir)
    print(df.loc[(df.row_type=="matched_seated_pair") & (df.window=="QRST") & (df.allow_polarity_reversal==False),
                 ["subject","size_mm","rho_ch1","rho_ch2","h1_minus_h2","block_delta_min","block_delta_max"]].to_string(index=False))


if __name__=="__main__":
    main()
