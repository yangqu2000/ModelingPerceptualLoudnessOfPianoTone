import librosa
import numpy as np
import pandas as pd
import pickle
import scipy.io
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

N_MELS = 8
GAP = 5

KEY_FAILURE_DICT_UPRIGHT = {
    21: (0, 25),
    38: (0, 25),
    39: (0, 36),
    40: (0, 30),
    45: (0, 26),
    46: (0, 32),
    51: (0, 31),
    55: (0, 25),
    72: (0, 66),
}

KEY_FAILURE_DICT_GRAND = {
    52: (112, 128),
    60: (109, 128),
    64: (99, 128),
    78: (121, 128),
}


def load(p, v, envir):
    wave = np.load(f'recording/{envir}-slice-npy/dictwave_{envir}_p={p}_v={v}.npy')
    if envir == 'upright':
        sr = 44100
    else:
        sr = 48000
    return librosa.resample(wave, orig_sr=sr, target_sr=22050)


def calculate_db_onset(p, v, envir):
    wave = load(p, v, envir=envir)

    S = librosa.feature.melspectrogram(y=wave, sr=22050, n_mels=8, hop_length=512)
    S_db = librosa.power_to_db(S, ref=1e-7)

    o_env = librosa.onset.onset_strength(y=wave, sr=22050)[:15]
    onset = np.argmax(o_env)

    # normalize to [0, 1]
    if envir == 'upright':
        S_db = (S_db - (-15.36853)) / (81.9082 + 15.36853)
    else:
        S_db = (S_db - (-8.088905)) / (96.59125 + 8.088905)

    return S_db, onset


def get_flattened_diff_D_db(ref_pitch, ref_vel, var_pitch, var_vel, envir):
    # ref_D_db, ref_onset = calculate_db_onset(ref_pitch, ref_vel, envir)
    # var_D_db, var_onset = calculate_db_onset(var_pitch, var_vel, envir)
    #
    # diff_D_db = var_D_db[:, var_onset:var_onset + GAP] - ref_D_db[:, ref_onset:ref_onset + GAP]

    ref_D_db = np.load(f"data/recordings/{envir}-slice-spec-npy/dictspec_{envir}_p={ref_pitch}_v={ref_vel}.npy")
    var_D_db = np.load(f"data/recordings/{envir}-slice-spec-npy/dictspec_{envir}_p={var_pitch}_v={var_vel}.npy")

    diff_D_db = var_D_db - ref_D_db

    nf, nt = diff_D_db.shape
    diff_D_db_flatten = np.reshape(diff_D_db, nf * nt)

    return diff_D_db_flatten


def calibration(envir):

    model = pickle.load(open(f'model/model_{envir}.pkl', 'rb'))
    coef = model.coef_.reshape(N_MELS, GAP)

    if envir == 'upright':        
        sine_loudness = scipy.io.loadmat('data/calibration/calibration_conferenceRoom.mat')
        sine_loudness = sine_loudness['sine_CR'].T
    else:
        sine_loudness = scipy.io.loadmat('data/calibration/calibration_studio.mat')
        sine_loudness = sine_loudness['sine_Stu'].T

    vel_range = list(range(1, 127, 8)) + [127]

    ml_sound_Is = []
    matlab_sound_Is = []
    for v_idx, v in enumerate(vel_range):
        # D_db, onset = calculate_db_onset(69, v, envir=envir)
        # after_db = D_db[:, onset:onset + GAP]
        after_db = np.load(f"data/recordings/{envir}-slice-spec-npy/dictspec_{envir}_p=69_v={v}.npy")
        ml_sound_Is.append(np.sum(after_db * coef))
        matlab_sound_Is.append(sine_loudness[v_idx][0])

    if envir == 'grand':
        ml_sound_Is = ml_sound_Is[:-2]
        matlab_sound_Is = matlab_sound_Is[:-2]
    else:
        ml_sound_Is = ml_sound_Is[3:-5]
        matlab_sound_Is = matlab_sound_Is[3:-5]

    ml2matlab = interpolate.interp1d(ml_sound_Is, matlab_sound_Is, fill_value='extrapolate')

    return ml2matlab


def generate_loudness_matrix(envir, model):

    coef = model.coef_.reshape(N_MELS, GAP)
    ml2matlab = calibration(envir=envir)

    sound_Is = []
    for p in range(21, 109):
        sound_I = []
        for v in range(0, 128):
            # D_db, onset = calculate_db_onset(p, v, envir=envir)
            # after_db = D_db[:, onset:onset + GAP]

            after_db = np.load(f"data/recordings/{envir}-slice-spec-npy/dictspec_{envir}_p={p}_v={v}.npy")
            loudness = np.sum(after_db * coef)
            loudness_calibrated = ml2matlab(loudness)
            sound_I.append(loudness_calibrated)
        sound_Is.append(sound_I)

    sound_Is = np.array(sound_Is)

    np.save(f'data/loudness_matrix/parametric_{envir}_loudness_matrix.npy', sound_Is)


# Code below is contributed by Avery (Hangkai Qian)
def compute_eq_loudness_line_formal(data, vel, ref_pitch=69):
    ref_sone = data[vel, ref_pitch]
    
    candidates = {}
    for pitch in range(0, 88):
        candidates[pitch] = []
        
        upper_range = 90 if pitch in [31, 39, 43, 57] else 128  # an easy way to avoid nan
        
        for v in range(-1, upper_range):
            if v == -1:
                s1, s2 = 0, data[v + 1, pitch]
            elif v == 127:
                s1, s2 = data[v, pitch], float('Inf')
            else:
                s1, s2 = data[v, pitch], data[v + 1, pitch]
            if min(s1, s2) <= ref_sone <= max(s1, s2):
                if v == -1:
                    closest_v = 0
                elif v == 127:
                    closest_v = 127
                else:
                    closest_v = v if np.abs(np.array([s1, s2], dtype=np.float64) - ref_sone).argmin() == 0 else v + 1
                if closest_v not in candidates[pitch]:
                    candidates[pitch].append(closest_v) 
    return candidates


def candidates_dict_to_candidates_array(candidates):
    cand_array = []
    for pitch in candidates.keys():
        cand_array += [[v, pitch] for v in candidates[pitch]]
    return np.array(cand_array)


def candicates_to_line1(candidates, ref_pitch=60, moving_average_lgth=25):
    # assert len(candidates[ref_pitch]) == 1
    
    ref_v = max(candidates[ref_pitch])
    
    line = np.zeros(88)
    line[ref_pitch] = ref_v
    
    for pitch in range(ref_pitch + 1, 88):
        cand_v = np.array(candidates[pitch])
        prev_vs = line[max(int(pitch - moving_average_lgth), ref_pitch): pitch]
        if len(cand_v)>0:
            line[pitch] = cand_v[np.abs(cand_v - prev_vs.mean()).argmin()]
        else:
            line[pitch] = np.nan

    for pitch in range(ref_pitch - 1, -1, -1):
        cand_v = np.array(candidates[pitch])
        prev_vs = line[pitch + 1: min(int(pitch + moving_average_lgth + 1), ref_pitch + 1)]
        if len(cand_v)>0:
            line[pitch] = cand_v[np.abs(cand_v - prev_vs.mean()).argmin()]
        else:
            line[pitch] = np.nan

    for i in range(len(line)):
        if np.isnan(line[i]):
            line[i] = (line[i-1]+line[i+1])/2
    return line


def candicates_to_line3(candidates, vel, ref_pitch=60, moving_average_lgth=15):
    # assert vel in candidates[ref_pitch]
    line = np.zeros(88)
    for pitch in range(88):
        line[pitch] = np.median(np.array(candidates[pitch]))
    line[ref_pitch] = vel

    return line


def plot_heatmap(model, notetype):
    ref_pitch = 48
    # colors = ['turquoise','gold', 'lawngreen', 'salmon']
    colors = ['white','white','white','white']
    loudness = np.load(f'data/loudness_matrix/{model}_{notetype}_loudness_matrix.npy').T
    
    figure = plt.figure(figsize=(25, 20))
    ax = plt.subplot(211)
    im = ax.imshow(loudness, origin='lower', aspect='auto', cmap='magma')
    
    for i, v in enumerate([32, 44, 60, 80]):
        cands = compute_eq_loudness_line_formal(loudness, v, ref_pitch)
        line = candicates_to_line1(cands, ref_pitch, 30)
        ax.plot(line, label=f'vel={v}', color=colors[i], linewidth=2, linestyle='--', zorder=2)
    
    plt.xlabel('MIDI Pitch', fontsize=40)
    plt.ylabel('MIDI Velocity', fontsize=40)
    
    ax.set_xticks(np.arange(0, 89, 12))
    ax.set_yticks(np.arange(0, 129, 16))
    ax.set_xticklabels(np.arange(21, 109, 12), fontsize=30)
    ax.set_yticklabels(np.arange(0, 129, 16), fontsize=30)
    
    # color bar
    cax = figure.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax)
    cbar.ax.tick_params(labelsize=30) 
    
    # light up failed notes
    if model == 'ISO-532-3' or model == 'intensity-based' or model == 'parametric':
        if notetype == 'upright':
            for i, v in KEY_FAILURE_DICT_UPRIGHT.items():
                ax.add_patch(Rectangle((i-21.5, v[0]), 1, v[1]-v[0]+1, fill=True, color='white', edgecolor='white', lw=2, linestyle='--', zorder=1, alpha=0.4))
        else:
            for i,v in KEY_FAILURE_DICT_GRAND.items():
                ax.add_patch(Rectangle((i-21.5, v[0]-1), 1, v[1]-v[0]+1, fill=True, color='white', edgecolor='white', lw=2, linestyle='--', zorder=1, alpha=0.7))
    
    plt.show()
            
        

