"""
Скрипты для первичной обработки данных с робота
"""
import joblib
from ml.image_model.image_model import BLIPImageCaptioner
from pathlib import Path
from pickle import load
import numpy as np
import math
import pandas as pd
import time
from scipy.signal import savgol_filter
from backend.models.odometry_data import OdometryData, OdometryDataFromBin


def preprocess_image(camera_output: list) -> str:
    """
    :param camera_output: image from robot camera
    :param lidar_output: lidar vectors from robot LiDaR
    :return:
    """
    image_model = BLIPImageCaptioner()
    if not camera_output:
        return ''
    return "Data from camera: " + image_model.invoke(camera_output[-1]) + "."


import numpy as np
import math
from scipy.signal import savgol_filter
from scipy import stats
from backend.models.odometry_data import OdometryData, OdometryDataFromBin


def convert_types(inp: dict) -> dict:
    return {
        "frame_id": inp['frame_id'],
        "timestamp": inp['timestamp'],
        'position_x': inp['position']['x'],
        'position_y': inp['position']['y'],
        'position_z': inp['position']['z'],
        'orientation_x': inp['orientation']['x'],
        'orientation_y': inp['orientation']['y'],
        'orientation_z': inp['orientation']['z'],
        'orientation_w': inp['orientation']['w'],
        'linear_velocity_x': inp['linear_velocity']['x'],
        'linear_velocity_y': inp['linear_velocity']['y'],
        'linear_velocity_z': inp['linear_velocity']['z'],
        'angular_velocity_x': inp['angular_velocity']['x'],
        'angular_velocity_y': inp['angular_velocity']['y'],
        'angular_velocity_z': inp['angular_velocity']['z']
    }


def compute_behavioral_indicators(data_dict):
    """
    Extracts 11 behavioral indicators with all requested improvements:
    - Adaptive speed normalization (by max in trajectory)
    - Fixed area_redundancy (avg_dist_to_center / total_distance)
    - 4 new indicators: oscillation_ratio, acceleration_stability, path_curvature, velocity_direction_coherence
    - angular_zcr normalized by measurement count (not time)

    Returns:
        dict with keys:
            'cleaned_avg_linear_speed',      # Normalized by max speed in THIS trajectory
            'speed_variance',
            'angular_zcr',                   # Ratio to measurement count
            'mid_stop_ratio',
            'zcr_angular_vel',
            'progress_consistency',
            'angular_smoothness',
            'area_redundancy',               # Fixed formula
            'oscillation_ratio',             # NEW
            'acceleration_stability',        # NEW
            'velocity_direction_coherence'   # NEW (path_curvature merged into angular_smoothness)
    """
    # === 1. Data extraction ===
    new_odometries = data_dict['odometries']
    odometries = new_odometries.copy()
    for i in range(len(new_odometries)):
        odometries[i] = convert_types(new_odometries[i])
    n_samples = len(odometries)

    if n_samples == 0:
        keys = [
            'cleaned_avg_linear_speed', 'speed_variance', 'angular_zcr',
            'mid_stop_ratio', 'zcr_angular_vel', 'progress_consistency',
            'angular_smoothness', 'area_redundancy', 'oscillation_ratio',
            'acceleration_stability', 'velocity_direction_coherence',
            'trajectory_entropy', 'hurst_exponent', 'jarque_bera_acceleration',
            'path_curvature', 'speed_coefficient_of_variation'
        ]
        return {k: 0.0 for k in keys}

    pos_x = np.empty(n_samples)
    pos_y = np.empty(n_samples)

    for i, odom in enumerate(odometries):
        pos_x[i] = float(odom.get('position_x', 0.0))
        pos_y[i] = float(odom.get('position_y', 0.0))

    # === 2. Sampling interval ===
    try:
        ts = np.array([
            float(odom['timestamp']) if isinstance(odom['timestamp'], str)
            else odom['timestamp'] for odom in odometries
        ])
        if ts.max() > 1e9:  # ms → s
            ts = ts / 1000.0
        time_diffs = np.diff(ts)
        valid_diffs = time_diffs[time_diffs > 0]
        sampling_interval = np.median(valid_diffs) if len(valid_diffs) > 0 else 0.5
        if sampling_interval <= 0:
            sampling_interval = 0.5
    except:
        sampling_interval = 0.5

    # === 3. Smoothing ===
    win = min(9, max(5, n_samples // 10))
    if n_samples >= win:
        try:
            pos_x_s = savgol_filter(pos_x, win, 2)
            pos_y_s = savgol_filter(pos_y, win, 2)
        except:
            k = np.ones(win) / win
            pos_x_s = np.convolve(pos_x, k, 'same')
            pos_y_s = np.convolve(pos_y, k, 'same')
            h = win // 2
            if h > 0:
                pos_x_s[:h] = pos_x[:h]
                pos_x_s[-h:] = pos_x[-h:]
                pos_y_s[:h] = pos_y[:h]
                pos_y_s[-h:] = pos_y[-h:]
    else:
        pos_x_s, pos_y_s = pos_x, pos_y

    # === 4. Kinematics ===
    vel_x = np.gradient(pos_x_s, sampling_interval)
    vel_y = np.gradient(pos_y_s, sampling_interval)
    speed_raw = np.sqrt(vel_x ** 2 + vel_y ** 2)

    # Denoise speed
    if n_samples >= 5:
        speed_diff = np.abs(np.diff(speed_raw))
        noise_level = np.percentile(speed_diff, 25) if speed_diff.size > 0 else 0.01
        speed_c = speed_raw.copy()
        r = min(2, n_samples // 20)
        for i in range(r, n_samples - r):
            w = speed_raw[i - r:i + r + 1]
            if np.std(w) > noise_level * 2:
                speed_c[i] = np.median(w)
    else:
        speed_c = speed_raw

    avg_speed = np.mean(speed_c) if n_samples > 0 else 0.0
    max_speed_in_trajectory = np.max(speed_c) if n_samples > 0 else 0.0

    # Precompute angles
    if n_samples > 1:
        dx = np.diff(pos_x_s)
        dy = np.diff(pos_y_s)
        angles = np.arctan2(dy, dx)
        unwrapped_angles = np.unwrap(angles)
    else:
        angles = np.array([])
        unwrapped_angles = np.array([])
        dx, dy = np.array([]), np.array([])

    # Forward speed for ZCR
    forward_speed = np.zeros(n_samples)
    if n_samples > 1:
        segment_lengths = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        forward_dir_x = dx / segment_lengths
        forward_dir_y = dy / segment_lengths
        for i in range(n_samples):
            if i == 0:
                vx, vy = vel_x[i], vel_y[i]
                fdx, fdy = forward_dir_x[0], forward_dir_y[0]
            elif i == n_samples - 1:
                vx, vy = vel_x[i], vel_y[i]
                fdx, fdy = forward_dir_x[-1], forward_dir_y[-1]
            else:
                vx, vy = vel_x[i], vel_y[i]
                fdx, fdy = forward_dir_x[i - 1], forward_dir_y[i - 1]
            forward_speed[i] = vx * fdx + vy * fdy

    indicators = {}

    # === 1. cleaned_avg_linear_speed (ADAPTIVE NORMALIZATION) ===
    adaptive_max_speed = max(0.1, max_speed_in_trajectory)
    indicators['cleaned_avg_linear_speed'] = np.clip(avg_speed / adaptive_max_speed, 0.0, 1.0)

    # === 2. speed_variance ===
    if n_samples > 1:
        spd_std = np.std(speed_c)
        reference = avg_speed * 0.8 + 0.1
        variance_norm = spd_std / (reference + 1e-8)
        indicators['speed_variance'] = 1.0 - np.clip(variance_norm, 0.0, 1.0)
    else:
        indicators['speed_variance'] = 1.0

    # === 3. angular_zcr (RATIO TO MEASUREMENTS COUNT) ===
    if n_samples > 3 and len(unwrapped_angles) >= 2:
        ang_vel = np.diff(unwrapped_angles) / sampling_interval
        zc_count = 0
        for i in range(1, len(ang_vel)):
            if ang_vel[i - 1] == 0 or ang_vel[i] == 0:
                continue
            if ang_vel[i - 1] * ang_vel[i] < 0:
                zc_count += 1

        total_measurements = len(ang_vel)
        zc_rate = zc_count / (total_measurements + 1e-8)
        indicators['angular_zcr'] = 1.0 - np.clip(zc_rate / 0.3, 0.0, 1.0)
    else:
        indicators['angular_zcr'] = 1.0

    # === 4. mid_stop_ratio ===
    if n_samples > 1:
        stopped = speed_c < 0.1
        start_idx = 0
        while start_idx < n_samples and stopped[start_idx]:
            start_idx += 1
        end_idx = n_samples - 1
        while end_idx >= 0 and stopped[end_idx]:
            end_idx -= 1
        if start_idx < end_idx:
            mid_stopped_count = np.sum(stopped[start_idx:end_idx + 1])
            mid_stop_ratio = mid_stopped_count / n_samples
        else:
            mid_stop_ratio = 0.0
        indicators['mid_stop_ratio'] = 1.0 - np.clip(mid_stop_ratio, 0.0, 1.0)
    else:
        indicators['mid_stop_ratio'] = 1.0

    # === 5. zcr_angular_vel ===
    if n_samples > 2:
        zc_count = 0
        for i in range(1, n_samples):
            if forward_speed[i - 1] == 0 or forward_speed[i] == 0:
                continue
            if forward_speed[i - 1] * forward_speed[i] < 0:
                zc_count += 1
        zc_rate = zc_count / max(1e-8, (n_samples - 1) * sampling_interval)
        indicators['zcr_angular_vel'] = 1.0 - np.clip(zc_rate / 2.0, 0.0, 1.0)
    else:
        indicators['zcr_angular_vel'] = 1.0

    # === 6. progress_consistency ===
    if n_samples >= 2:
        path_len = 0.0
        for i in range(n_samples - 1):
            path_len += math.dist((pos_x[i], pos_y[i]), (pos_x[i + 1], pos_y[i + 1]))
        final_len = math.dist((pos_x[0], pos_y[0]), (pos_x[-1], pos_y[-1]))
        consistency = final_len / (path_len + 1e-8) if path_len > 0.02 else 0.0
        indicators['progress_consistency'] = np.clip(consistency, 0.0, 1.0)
    else:
        indicators['progress_consistency'] = 0.0

    # === 7. angular_smoothness ===
    if n_samples > 4 and len(unwrapped_angles) >= 3:
        ang_vel = np.diff(unwrapped_angles) / sampling_interval
        delta_ang_vel = np.abs(np.diff(ang_vel))
        if len(delta_ang_vel) > 0:
            threshold = np.percentile(delta_ang_vel, 75) + 0.5 * np.std(delta_ang_vel)
            rough_ratio = np.mean(delta_ang_vel > threshold)
            smoothness = 1.0 - rough_ratio
            indicators['angular_smoothness'] = np.clip(smoothness, 0.0, 1.0)
        else:
            indicators['angular_smoothness'] = 1.0
    else:
        indicators['angular_smoothness'] = 1.0

    # === 8. area_redundancy (FIXED AS REQUESTED) ===
    if n_samples >= 3:
        # Center of mass
        mean_x = np.mean(pos_x_s)
        mean_y = np.mean(pos_y_s)

        # Average distance to center
        avg_dist_to_center = 0.0
        for i in range(n_samples):
            avg_dist_to_center += math.dist((pos_x_s[i], pos_y_s[i]), (mean_x, mean_y))
        avg_dist_to_center /= n_samples

        # Total distance traveled
        total_distance = 0.0
        for i in range(n_samples - 1):
            total_distance += math.dist((pos_x_s[i], pos_y_s[i]), (pos_x_s[i + 1], pos_y_s[i + 1]))

        # Ratio: for straight line ≈ 0.25, for circling ≈ 0.02-0.05
        ratio = avg_dist_to_center / (total_distance + 1e-8)

        # Normalize: 0.25 = max for normal trajectories
        max_ratio = 0.25
        area_redundancy = ratio / max_ratio
        indicators['area_redundancy'] = np.clip(area_redundancy, 0.0, 1.0)
    else:
        indicators['area_redundancy'] = 1.0

    # === 9. oscillation_ratio (NEW) ===
    if n_samples >= 15:
        end_idx = max(10, int(0.8 * n_samples))  # Last 20% of trajectory
        final_position = (pos_x_s[-1], pos_y_s[-1])
        oscillations = 0

        for i in range(end_idx, n_samples - 1):
            current_dist = math.dist((pos_x_s[i], pos_y_s[i]), final_position)
            next_dist = math.dist((pos_x_s[i + 1], pos_y_s[i + 1]), final_position)
            # Count oscillation if robot got closer then moved away significantly
            if current_dist < 0.3 and next_dist > current_dist * 1.2:
                oscillations += 1

        oscillation_ratio = oscillations / (n_samples - end_idx + 1e-8)
        # Scale: 1 oscillation per 3 points = 0.33 → 0.0 after scaling
        indicators['oscillation_ratio'] = 1.0 - np.clip(oscillation_ratio * 3, 0.0, 1.0)
    else:
        indicators['oscillation_ratio'] = 1.0

    # === 10. acceleration_stability (NEW) ===
    if n_samples > 3:
        acceleration = np.gradient(speed_c, sampling_interval)
        max_accel = np.max(np.abs(acceleration)) if acceleration.size > 0 else 0.0
        # Physical limit for small robots: 0.8 m/s²
        indicators['acceleration_stability'] = 1.0 - np.clip(max_accel / 0.8, 0.0, 1.0)
    else:
        indicators['acceleration_stability'] = 1.0

    # === 11. velocity_direction_coherence (NEW) ===
    if n_samples > 2:
        coherence_count = 0
        total_count = 0

        for i in range(n_samples - 1):
            if speed_c[i] < 0.05:  # Skip stops
                continue

            total_count += 1

            # Movement direction from positions
            move_dir_x = pos_x_s[i + 1] - pos_x_s[i]
            move_dir_y = pos_y_s[i + 1] - pos_y_s[i]
            move_norm = math.sqrt(move_dir_x ** 2 + move_dir_y ** 2) + 1e-8
            move_dir_x /= move_norm
            move_dir_y /= move_norm

            # Direction from velocity
            vel_norm = speed_c[i] + 1e-8
            vel_dir_x = vel_x[i] / vel_norm
            vel_dir_y = vel_y[i] / vel_norm

            # Angle between directions
            cos_angle = move_dir_x * vel_dir_x + move_dir_y * vel_dir_y
            angle = math.acos(max(-1.0, min(1.0, cos_angle)))

            # Coherent if angle < 45 degrees
            if angle < math.pi / 4:
                coherence_count += 1

        ratio = coherence_count / (total_count + 1e-8) if total_count > 0 else 1.0
        indicators['velocity_direction_coherence'] = np.clip(ratio, 0.0, 1.0)
    else:
        indicators['velocity_direction_coherence'] = 1.0

    # === 12. trajectory_entropy (NEW) ===
    if n_samples >= 5:
        # Adaptive grid size based on trajectory dimensions
        x_min, x_max = np.min(pos_x_s), np.max(pos_x_s)
        y_min, y_max = np.min(pos_y_s), np.max(pos_y_s)
        x_range = max(x_max - x_min, 0.1)
        y_range = max(y_max - y_min, 0.1)

        cell_size_x = x_range / 10  # 10 cells in x direction
        cell_size_y = y_range / 10  # 10 cells in y direction

        # Count visits per cell
        cell_counts = {}
        for i in range(n_samples):
            cell_x = int((pos_x_s[i] - x_min) / cell_size_x)
            cell_y = int((pos_y_s[i] - y_min) / cell_size_y)
            cell_key = (cell_x, cell_y)
            cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1

        # Calculate probabilities and entropy
        total_points = n_samples
        entropy = 0.0
        for count in cell_counts.values():
            p = count / total_points
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = math.log2(min(100, total_points))  # max 100 cells
        trajectory_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        indicators['trajectory_entropy'] = 1.0 - np.clip(trajectory_entropy, 0.0, 1.0)
    else:
        indicators['trajectory_entropy'] = 1.0

    # === 13. hurst_exponent (NEW) ===
    if n_samples >= 10 and avg_speed > 0.01:
        # R/S analysis for Hurst exponent
        cum_dev = np.cumsum(speed_c - avg_speed)
        r_s_values = []
        window_sizes = [5, 10, 20, 30]  # Different window sizes

        for window in window_sizes:
            if window >= n_samples:
                continue

            r_s_ratios = []
            for i in range(0, n_samples - window + 1, window):
                segment = cum_dev[i:i + window]
                if len(segment) < window:
                    continue

                r = np.max(segment) - np.min(segment)
                s = np.std(speed_c[i:i + window]) + 1e-8
                r_s_ratios.append(r / s)

            if r_s_ratios:
                r_s_values.append((window, np.mean(r_s_ratios)))

        if len(r_s_values) >= 2:
            log_sizes = np.log([x[0] for x in r_s_values])
            log_rs = np.log([x[1] for x in r_s_values])
            slope, intercept = np.polyfit(log_sizes, log_rs, 1)
            hurst = slope

            # Normalize: 0.5 = random walk, 0.0-0.5 = mean-reverting, 0.5-1.0 = trending
            hurst_norm = 1.0 - abs(hurst - 0.5) * 2
            indicators['hurst_exponent'] = np.clip(hurst_norm, 0.0, 1.0)
        else:
            indicators['hurst_exponent'] = 0.5
    else:
        indicators['hurst_exponent'] = 0.5

    # === 14. jarque_bera_acceleration (NEW) ===
    if n_samples >= 5:
        acceleration = np.gradient(speed_c, sampling_interval) if n_samples > 2 else np.array([])
        if len(acceleration) >= 4:
            # Jarque-Bera test for normality of acceleration distribution
            try:
                jb_stat, p_value = stats.jarque_bera(acceleration)

                # Critical value for chi-square with 2 df at 0.05 significance level
                critical_value = 5.991

                # Higher JB statistic means more non-normal (anomalous)
                jb_ratio = jb_stat / critical_value
                jarque_bera_score = 1.0 - min(jb_ratio, 1.0)
                indicators['jarque_bera_acceleration'] = jarque_bera_score
            except:
                indicators['jarque_bera_acceleration'] = 1.0
        else:
            indicators['jarque_bera_acceleration'] = 1.0
    else:
        indicators['jarque_bera_acceleration'] = 1.0

    # === 15. path_curvature (NEW) ===
    if n_samples >= 5:
        # Calculate curvature using finite differences
        curvatures = []
        for i in range(1, n_samples - 1):
            # First derivatives (velocity)
            dx1 = (pos_x_s[i] - pos_x_s[i - 1]) / sampling_interval
            dy1 = (pos_y_s[i] - pos_y_s[i - 1]) / sampling_interval

            # Second derivatives (acceleration)
            dx2 = (pos_x_s[i + 1] - 2 * pos_x_s[i] + pos_x_s[i - 1]) / (sampling_interval ** 2)
            dy2 = (pos_y_s[i + 1] - 2 * pos_y_s[i] + pos_y_s[i - 1]) / (sampling_interval ** 2)

            # Curvature formula
            curvature = abs(dx1 * dy2 - dy1 * dx2) / (dx1 ** 2 + dy1 ** 2 + 1e-8) ** 1.5
            curvatures.append(curvature)

        if curvatures:
            avg_curvature = np.mean(curvatures)
            # Normalize: 0.1 is typical max for normal robot motion
            curvature_norm = 1.0 - min(avg_curvature / 0.1, 1.0)
            indicators['path_curvature'] = np.clip(curvature_norm, 0.0, 1.0)
        else:
            indicators['path_curvature'] = 1.0
    else:
        indicators['path_curvature'] = 1.0

    # === 16. speed_coefficient_of_variation (NEW) ===
    if n_samples > 1 and avg_speed > 0.01:
        cv = np.std(speed_c) / avg_speed
        # Inverted: lower CV = more stable speed = higher score
        cv_norm = 1.0 - min(cv, 1.0)
        indicators['speed_coefficient_of_variation'] = np.clip(cv_norm, 0.0, 1.0)
    else:
        indicators['speed_coefficient_of_variation'] = 1.0

    # Final safety check
    for key in indicators:
        indicators[key] = float(np.clip(indicators[key], 0.0, 1.0))

    return indicators


def _norm(value, min_val, max_val):
    """Fast normalization to [0, 1] range"""
    if max_val <= min_val:
        return 0.0
    norm_val = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, norm_val))


def check_for_idle(odometry) -> bool:
    for pon in odometry:
        linear_velocity = abs(pon['linear_velocity']['x']) + abs(pon['linear_velocity']['y']) + abs(
            pon['linear_velocity']['z'])
        angular_velocity = abs(pon['angular_velocity']['x']) + abs(pon['angular_velocity']['y']) + abs(
            pon['angular_velocity']['z'])
        if linear_velocity + angular_velocity > 0.01:
            return False
    return True


def preprocess_data(output) -> str:
    """
    Transforms raw data into image/text pair for VLM
    :param output:
    :return:
    """
    camera_output = output['images']
    odometry = output['odometries']

    if not odometry:
        return ''

    # getting useful features for models
    start_time = time.time_ns()
    image_data = preprocess_image(camera_output)
    pic_time = time.time_ns()
    odom_data = compute_behavioral_indicators(output)
    indicator_time = time.time_ns()
    odom_data = pd.DataFrame(data=[odom_data.values()], columns=list(odom_data.keys()))

    if check_for_idle(odometry) and 0:
        anomaly_type = 'Маршрут не найден, робот стоит на месте'
    else:
        # loading classification model
        pkl_path = Path(__file__).parent / 'classifier.pkl'
        with open(pkl_path, 'rb') as f:
            classifier = load(f)
        print('Indicators:', odom_data)
        anomaly_type = classifier.predict_proba(odom_data)[0, 1]
        print('Probability:', anomaly_type)
        threshold = 0.6
        if anomaly_type < threshold:
            return ''
        anomaly_type = 'Аномальное движение, робот сбит с толку'
    anom_data = f'Вердикт системы аналитики аномалий: {anomaly_type}.'

    classificator_time = time.time_ns()

    print(f'Pic time:           \t\t\t{(pic_time - start_time) * 1e-9:.3f}')
    print(f'Indicator time:     \t\t\t{(indicator_time - pic_time) * 1e-9:.3f}')
    print(f'Classificator time: \t\t\t{(classificator_time - indicator_time) * 1e-9:.3f}')

    return anom_data + image_data


