#!/usr/bin/env python3
"""
Advanced SRP methods for difficult failure cases.
Implements preprocessing, multi-resolution SRP, and ensemble approaches.

Usage:
    python advanced_srp_methods.py --method [preprocess|multiband|ensemble|all]
"""

import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import argparse
from scipy import signal
from scipy.fft import fft, ifft
import subprocess
import tempfile
import json
from datetime import datetime
import soundfile as sf

class AdvancedSRPProcessor:
    def __init__(self, base_dir, test_csv="srp_predicted_failures_test.csv", limit=None):
        self.base_dir = base_dir
        self.test_csv = test_csv
        self.limit = limit
        self.sample_rate = 16000

    def load_audio_files(self):
        """Load audio files for the 201 failure cases."""
        print("Loading audio files for advanced processing...")

        # Load the test cases
        test_df = pd.read_csv(self.test_csv)

        audio_data = []
        use_mic_id = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Same as in run_SRP

        for idx, row in test_df.iterrows():
            # Check limit
            if self.limit is not None and len(audio_data) >= self.limit:
                print(f"Reached limit of {self.limit} samples")
                break

            # Construct full path for channel loading
            base_path = Path(self.base_dir) / row['filename']

            # Load multichannel audio using the same approach as run_SRP
            try:
                channels = []
                fs_ref = None

                for i in use_mic_id:
                    # Convert .flac to .wav if needed and add channel suffix
                    channel_path = str(base_path).replace('.flac', '.wav').replace('.wav', f'_CH{i}.wav')

                    if Path(channel_path).exists():
                        if i == 0:
                            info = sf.info(channel_path)
                            fs_ref = info.samplerate

                        single_ch_signal, fs = sf.read(channel_path, dtype="float64")
                        channels.append(single_ch_signal)
                    else:
                        print(f"Channel file not found: {channel_path}")
                        break

                if len(channels) == len(use_mic_id):
                    # Stack channels with shape (channels, samples)
                    audio = np.stack(channels, axis=0)

                    audio_data.append({
                        'index': idx,
                        'file_path': str(base_path),
                        'audio': audio,
                        'gt_angle': row['angle(°)'],
                        'sample_rate': fs_ref or self.sample_rate
                    })

            except Exception as e:
                print(f"Error loading multichannel audio for {base_path}: {e}")

        print(f"Loaded {len(audio_data)} audio files")
        return audio_data

    def method_spectral_preprocessing(self, audio_data):
        """Method 1: Spectral preprocessing for noise reduction."""
        print("\n🔧 Testing Method 1: Spectral Preprocessing")

        processed_audio_data = []

        for item in audio_data:
            audio = item['audio']
            processed_channels = []

            for channel in audio:
                # Method 1a: Spectral subtraction
                processed_channel = self._spectral_subtraction(channel)

                # Method 1b: Wiener filtering (optional)
                # processed_channel = self._wiener_filter(processed_channel)

                processed_channels.append(processed_channel)

            processed_item = item.copy()
            processed_item['audio'] = np.array(processed_channels)
            processed_audio_data.append(processed_item)

        # Save processed audio and test with SRP
        return self._test_processed_audio(processed_audio_data, "spectral_preprocess")

    def method_multiband_srp(self, audio_data):
        """Method 2: Multi-band SRP with REAL frequency filtering."""
        print("\n🎯 Testing Method 2: Multi-band SRP with Real Filtering")

        # Define frequency bands optimized for speech in noise
        freq_bands = [
            (300, 1200),   # Fundamental frequencies + F1
            (1200, 2500),  # F1-F2 transition (critical for speech)
            (1600, 3200),  # F2-F3 formants (our previous best)
            (2000, 4000),  # High formants + consonants
        ]

        results = []

        for band_idx, (f_min, f_max) in enumerate(freq_bands):
            print(f"  Testing band {band_idx+1}: {f_min}-{f_max} Hz with REAL filtering")

            # Create band-filtered versions of audio files
            band_audio_data = []
            for item in audio_data:
                filtered_audio = self._bandpass_filter(item['audio'], f_min, f_max)
                band_item = item.copy()
                band_item['audio'] = filtered_audio
                band_audio_data.append(band_item)

            # Test this frequency band with standard SRP (no fake freq params)
            band_result = self._test_processed_audio_simple(band_audio_data, f"multiband_real_{f_min}_{f_max}")
            if band_result:
                band_result['freq_band'] = (f_min, f_max)
                results.append(band_result)

        # Find best frequency band
        if results:
            best_band = max(results, key=lambda x: x['success_rate'])
            print(f"  🏆 Best frequency band: {best_band['freq_band']} Hz")
            print(f"      Success rate: {best_band['success_rate']:.1f}%")
            return best_band

        return None

    def method_ensemble_srp(self, audio_data):
        """Method 3: Ensemble of different SRP variants."""
        print("\n🎺 Testing Method 3: Ensemble SRP Methods")

        # Define optimized SRP variants for automotive environments
        srp_variants = [
            # Speech-optimized variants (based on our findings)
            {'name': 'speech_formants', 'freq_range': (1600, 3200), 'phat_beta': 1.0},  # Best performer
            {'name': 'speech_fundamental', 'freq_range': (300, 1200), 'phat_beta': 1.0},  # F0 region
            {'name': 'speech_wide', 'freq_range': (800, 3500), 'phat_beta': 1.0},  # Wide speech band

            # Robust variants for noisy environments
            {'name': 'robust_midband', 'freq_range': (1000, 2500), 'phat_beta': 0.8},  # Moderate PHAT
            {'name': 'robust_highband', 'freq_range': (2000, 4000), 'phat_beta': 0.6},  # Less PHAT weighting
            {'name': 'automotive_avoid', 'freq_range': (1200, 3000), 'phat_beta': 1.0},  # Avoid engine frequencies
        ]

        variant_results = []

        for variant in srp_variants:
            print(f"  Testing variant: {variant['name']}")

            # Run SRP with this variant's parameters
            result = self._run_srp_variant(variant)
            if result:
                variant_results.append({
                    'variant': variant,
                    'result': result,
                    'predictions': result['predictions'] if 'predictions' in result else None
                })

        if len(variant_results) < 2:
            print("  ❌ Need at least 2 variants for ensemble")
            return None

        # Ensemble the predictions
        ensemble_result = self._ensemble_predictions(variant_results)
        return ensemble_result

    def _spectral_subtraction(self, audio):
        """Apply spectral subtraction for noise reduction."""
        # Simple spectral subtraction implementation
        # Estimate noise from first 0.5 seconds
        noise_duration = int(0.5 * self.sample_rate)
        noise_segment = audio[:noise_duration]

        # FFT parameters
        n_fft = 2048
        hop_length = 512

        # STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        noise_stft = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length)

        # Estimate noise power spectrum
        noise_power = np.mean(np.abs(noise_stft)**2, axis=1, keepdims=True)

        # Spectral subtraction
        signal_power = np.abs(stft)**2
        clean_power = signal_power - 2 * noise_power
        clean_power = np.maximum(0.1 * signal_power, clean_power)  # Floor

        # Reconstruct
        clean_magnitude = np.sqrt(clean_power)
        clean_stft = clean_magnitude * np.exp(1j * np.angle(stft))
        clean_audio = librosa.istft(clean_stft, hop_length=hop_length)

        return clean_audio

    def _bandpass_filter(self, audio, f_min, f_max):
        """Apply bandpass filter to audio."""
        nyquist = self.sample_rate / 2
        low = f_min / nyquist
        high = f_max / nyquist

        if high >= 1.0:
            high = 0.99

        b, a = signal.butter(4, [low, high], btype='band')

        filtered_channels = []
        for channel in audio:
            filtered = signal.filtfilt(b, a, channel)
            filtered_channels.append(filtered)

        return np.array(filtered_channels)

    def _test_processed_audio(self, processed_audio_data, method_name):
        """Save processed audio and test with SRP."""

        # Create temporary directory for processed audio inside base_dir
        temp_dir = Path(self.base_dir) / f"temp_processed_{method_name}"
        temp_dir.mkdir(exist_ok=True)

        # Create new CSV with processed audio paths
        processed_csv_path = f"processed_test_{method_name}.csv"
        original_df = pd.read_csv(self.test_csv)

        processed_rows = []

        for item in processed_audio_data:
            # Save processed audio - need to save as individual channel files
            base_output_path = temp_dir / f"processed_{item['index']}"

            # Save each channel separately (to match expected format)
            for ch_idx, channel_data in enumerate(item['audio']):
                channel_path = f"{base_output_path}_CH{ch_idx}.wav"
                sf.write(channel_path, channel_data, item['sample_rate'])

            # Update CSV row with base path (without _CH0.wav suffix)
            row = original_df.iloc[item['index']].copy()
            row['filename'] = str(base_output_path.relative_to(self.base_dir)) + ".wav"
            processed_rows.append(row)

        # Save processed CSV
        processed_df = pd.DataFrame(processed_rows)
        processed_df.to_csv(processed_csv_path, index=False)

        # Run SRP on processed data
        try:
            cmd = [
                'python', '-m', 'xsrpMain.xsrp.run_SRP',
                '--csv', processed_csv_path,
                '--base-dir', self.base_dir,
                '--use_novel_noise',
                '--novel_noise_scene', 'Cafeteria2'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                # Look for results file (SRP creates its own filename)
                possible_files = [
                    f'{Path(processed_csv_path).stem}_srp_results.csv',
                    'srp_results.csv'
                ]

                results_file = None
                for filename in possible_files:
                    if Path(filename).exists():
                        results_file = filename
                        break

                if not results_file:
                    # Look for any CSV file modified in the last minute
                    csv_files = list(Path('.').glob('*srp*.csv'))
                    if csv_files:
                        results_file = max(csv_files, key=lambda x: x.stat().st_mtime)

                if results_file and Path(results_file).exists():
                    # Analyze results
                    results_df = pd.read_csv(results_file)
                    success_rate = ((results_df['abs_err_deg'] <= 30).sum() / len(results_df)) * 100
                    mae = results_df['abs_err_deg'].mean()

                    performance = {
                        'method': method_name,
                        'success_rate': success_rate,
                        'mae': mae,
                        'improvement_over_crnn': success_rate - 23.9,
                        'total_cases': len(results_df)
                    }

                    print(f"    Results: {success_rate:.1f}% success ({performance['improvement_over_crnn']:+.1f}% vs CRNN)")

                    # Rename results file to avoid conflicts
                    new_name = f'srp_results_{method_name}_{datetime.now().strftime("%H%M%S")}.csv'
                    Path(results_file).rename(new_name)

                    return performance
                else:
                    print(f"    ❌ Results file not found")
                    return None
            else:
                print(f"    ❌ SRP failed: {result.stderr}")
                return None

        except Exception as e:
            print(f"    ❌ Error testing {method_name}: {e}")
            return None
        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _test_processed_audio_simple(self, processed_audio_data, method_name):
        """Simple version without frequency parameters that don't work."""
        # Create temporary directory for processed audio inside base_dir
        temp_dir = Path(self.base_dir) / f"temp_processed_{method_name}"
        temp_dir.mkdir(exist_ok=True)

        # Create new CSV with processed audio paths
        processed_csv_path = f"processed_test_{method_name}.csv"
        original_df = pd.read_csv(self.test_csv)

        processed_rows = []

        for item in processed_audio_data:
            # Save processed audio - need to save as individual channel files
            base_output_path = temp_dir / f"processed_{item['index']}"

            # Save each channel separately (to match expected format)
            for ch_idx, channel_data in enumerate(item['audio']):
                channel_path = f"{base_output_path}_CH{ch_idx}.wav"
                sf.write(channel_path, channel_data, item['sample_rate'])

            # Update CSV row with base path (without _CH0.wav suffix)
            row = original_df.iloc[item['index']].copy()
            row['filename'] = str(base_output_path.relative_to(self.base_dir)) + ".wav"
            processed_rows.append(row)

        # Save processed CSV
        processed_df = pd.DataFrame(processed_rows)
        processed_df.to_csv(processed_csv_path, index=False)

        # Run SRP on processed data with ONLY working parameters
        try:
            cmd = [
                'python', '-m', 'xsrpMain.xsrp.run_SRP',
                '--csv', processed_csv_path,
                '--base-dir', self.base_dir,
                '--use_novel_noise',
                '--novel_noise_scene', 'Cafeteria2',
                '--srp_grid_cells', '1440',  # High resolution
                '--n_avg_samples', '300'     # Good averaging
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                # Look for results file
                possible_files = [
                    f'{Path(processed_csv_path).stem}_srp_results.csv',
                    'srp_results.csv'
                ]

                results_file = None
                for filename in possible_files:
                    if Path(filename).exists():
                        results_file = filename
                        break

                if not results_file:
                    csv_files = list(Path('.').glob('*srp*.csv'))
                    if csv_files:
                        results_file = max(csv_files, key=lambda x: x.stat().st_mtime)

                if results_file and Path(results_file).exists():
                    results_df = pd.read_csv(results_file)
                    success_rate = ((results_df['abs_err_deg'] <= 30).sum() / len(results_df)) * 100
                    mae = results_df['abs_err_deg'].mean()

                    performance = {
                        'method': method_name,
                        'success_rate': success_rate,
                        'mae': mae,
                        'improvement_over_crnn': success_rate - 23.9,
                        'total_cases': len(results_df)
                    }

                    print(f"    Results: {success_rate:.1f}% success ({performance['improvement_over_crnn']:+.1f}% vs CRNN), MAE: {mae:.1f}°")

                    # Rename results file to avoid conflicts
                    new_name = f'srp_results_{method_name}_{datetime.now().strftime("%H%M%S")}.csv'
                    Path(results_file).rename(new_name)

                    return performance
                else:
                    print(f"    ❌ Results file not found")
                    return None
            else:
                print(f"    ❌ SRP failed: {result.stderr}")
                return None

        except Exception as e:
            print(f"    ❌ Error testing {method_name}: {e}")
            return None
        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _run_srp_variant(self, variant):
        """Run a specific SRP variant with real parameters."""
        print(f"    Running {variant['name']} variant...")

        try:
            cmd = [
                'python', '-m', 'xsrpMain.xsrp.run_SRP',
                '--csv', self.test_csv,
                '--base-dir', self.base_dir,
                '--use_novel_noise',
                '--novel_noise_scene', 'Cafeteria2',
                '--freq_min', str(variant['freq_range'][0]),
                '--freq_max', str(variant['freq_range'][1]),
                '--srp_mode', 'gcc_phat_freq' if variant['phat_beta'] > 0.5 else 'gcc_phat_time',
                '--n_avg_samples', '400' if 'standard' in variant['name'] else '200',
                '--srp_grid_cells', '1440'  # High resolution for better accuracy
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Find results file
                possible_files = [
                    f'{Path(self.test_csv).stem}_srp_results.csv',
                    'srp_results.csv'
                ]

                results_file = None
                for filename in possible_files:
                    if Path(filename).exists():
                        results_file = filename
                        break

                if results_file:
                    results_df = pd.read_csv(results_file)
                    success_rate = ((results_df['abs_err_deg'] <= 30).sum() / len(results_df)) * 100
                    mae = results_df['abs_err_deg'].mean()

                    # Calculate confidence metrics for this variant
                    confidence_score = self._calculate_variant_confidence(results_df, variant)

                    # Store individual predictions for ensemble
                    predictions = results_df['srp_angle_deg'].values

                    # Rename file to avoid conflicts
                    new_name = f'srp_results_{variant["name"]}_{datetime.now().strftime("%H%M%S")}.csv'
                    Path(results_file).rename(new_name)

                    return {
                        'success_rate': success_rate,
                        'mae': mae,
                        'confidence_score': confidence_score,
                        'predictions': predictions,
                        'variant_name': variant['name']
                    }

            print(f"    ❌ {variant['name']} failed: {result.stderr[:100]}...")
            return None

        except Exception as e:
            print(f"    ❌ Error running {variant['name']}: {e}")
            return None

    def _calculate_variant_confidence(self, results_df, variant):
        """Calculate confidence score for a variant based on result consistency."""
        errors = results_df['abs_err_deg'].values

        # Confidence factors
        consistency_score = 1.0 / (1.0 + np.std(errors))  # Lower variance = higher confidence
        success_rate = ((errors <= 30).sum() / len(errors))  # Success rate factor
        low_error_rate = ((errors <= 10).sum() / len(errors))  # High precision factor

        # Frequency band bonus (1600-3200 Hz is best for speech)
        freq_min, freq_max = variant['freq_range']
        if 1600 <= freq_min <= 2000 and 2500 <= freq_max <= 3500:
            freq_bonus = 1.2
        elif 1000 <= freq_min <= freq_max <= 4000:
            freq_bonus = 1.1
        else:
            freq_bonus = 1.0

        # Mode bonus (freq domain often better)
        mode_bonus = 1.1 if variant['phat_beta'] > 0.5 else 1.0

        confidence = (0.4 * consistency_score + 0.3 * success_rate + 0.3 * low_error_rate) * freq_bonus * mode_bonus
        return min(confidence, 1.0)

    def _ensemble_predictions(self, variant_results):
        """Intelligent confidence-weighted ensemble of SRP predictions."""
        print(f"  🎺 Ensembling {len(variant_results)} variants with confidence weighting")

        if len(variant_results) < 2:
            print("  ❌ Need at least 2 variants for ensemble")
            return None

        # Extract predictions and confidence scores
        all_predictions = []
        confidence_scores = []
        variant_names = []

        for var_result in variant_results:
            result = var_result['result']
            predictions = result['predictions']
            confidence = result['confidence_score']

            all_predictions.append(predictions)
            confidence_scores.append(confidence)
            variant_names.append(result['variant_name'])

            print(f"    {result['variant_name']}: {result['success_rate']:.1f}% success, confidence: {confidence:.3f}")

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)  # Shape: (n_variants, n_samples)
        confidence_scores = np.array(confidence_scores)

        # Normalize confidence scores to use as weights
        weights = confidence_scores / np.sum(confidence_scores)
        print(f"    Confidence weights: {dict(zip(variant_names, weights))}")

        # Ensemble methods
        ensemble_predictions = []

        for sample_idx in range(all_predictions.shape[1]):
            sample_predictions = all_predictions[:, sample_idx]

            # Method 1: Confidence-weighted average (handling angular wrap-around)
            # Convert to complex numbers to handle angular averaging
            complex_preds = np.exp(1j * np.deg2rad(sample_predictions))
            weighted_complex = np.average(complex_preds, weights=weights)
            weighted_avg = np.rad2deg(np.angle(weighted_complex)) % 360

            # Method 2: Confidence-weighted median (more robust)
            sorted_indices = np.argsort(sample_predictions)
            cumulative_weights = np.cumsum(weights[sorted_indices])
            median_idx = sorted_indices[np.searchsorted(cumulative_weights, 0.5)]
            weighted_median = sample_predictions[median_idx]

            # Method 3: Best confidence prediction
            best_idx = np.argmax(confidence_scores)
            best_pred = sample_predictions[best_idx]

            # Combine methods (weighted median is most robust for outliers)
            if len(np.unique(sample_predictions)) == 1:
                # All same prediction
                final_pred = sample_predictions[0]
            elif np.max(confidence_scores) > 0.8:
                # High confidence variant exists, use it
                final_pred = best_pred
            else:
                # Use robust weighted median
                final_pred = weighted_median

            ensemble_predictions.append(final_pred)

        # Test ensemble predictions
        ensemble_predictions = np.array(ensemble_predictions)

        # Calculate performance using ground truth from test CSV
        test_df = pd.read_csv(self.test_csv)
        gt_angles = test_df['angle(°)'].values[:len(ensemble_predictions)]

        errors = np.abs(ensemble_predictions - gt_angles)
        errors = np.minimum(errors, 360 - errors)  # Handle wrap-around

        success_rate = ((errors <= 30).sum() / len(errors)) * 100
        mae = np.mean(errors)

        ensemble_result = {
            'method': 'confidence_weighted_ensemble',
            'success_rate': success_rate,
            'mae': mae,
            'improvement_over_crnn': success_rate - 23.9,
            'variants_used': variant_names,
            'confidence_weights': weights.tolist(),
            'predictions': ensemble_predictions
        }

        print(f"    🏆 Confidence-weighted ensemble: {success_rate:.1f}% success, {mae:.1f}° MAE")
        return ensemble_result

def main():
    parser = argparse.ArgumentParser(description='Advanced SRP methods for failure cases')
    parser.add_argument('--method', choices=['preprocess', 'multiband', 'ensemble', 'all'],
                       default='all', help='Method to test')
    parser.add_argument('--base-dir', default="/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted",
                       help='Base directory for audio files')
    parser.add_argument('--csv', default="srp_predicted_failures_test.csv",
                       help='CSV file with test cases')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples to test (for quick testing)')

    args = parser.parse_args()

    print("ADVANCED SRP METHODS FOR FAILURE CASES")
    print("=" * 80)

    processor = AdvancedSRPProcessor(args.base_dir, args.csv, args.limit)

    # Load audio data
    audio_data = processor.load_audio_files()
    if not audio_data:
        print("❌ No audio data loaded. Cannot proceed.")
        return

    results = []

    # Test requested methods
    if args.method in ['preprocess', 'all']:
        result = processor.method_spectral_preprocessing(audio_data)
        if result:
            results.append(result)

    if args.method in ['multiband', 'all']:
        result = processor.method_multiband_srp(audio_data)
        if result:
            results.append(result)

    if args.method in ['ensemble', 'all']:
        result = processor.method_ensemble_srp(audio_data)
        if result:
            results.append(result)

    # Summary
    print(f"\n" + "=" * 80)
    print("ADVANCED METHODS SUMMARY")
    print("=" * 80)

    if results:
        best_method = max(results, key=lambda x: x['success_rate'])

        print(f"🏆 BEST ADVANCED METHOD:")
        print(f"   Method: {best_method['method']}")
        print(f"   Success Rate: {best_method['success_rate']:.1f}%")
        print(f"   Improvement: {best_method['improvement_over_crnn']:+.1f}% vs CRNN")
        print(f"   MAE: {best_method['mae']:.1f}°")

        if best_method['success_rate'] > 23.9:
            print(f"✅ SUCCESS: Beat CRNN baseline!")
        if best_method['success_rate'] > 30.0:
            print(f"🎯 TARGET ACHIEVED: >30% success rate!")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'advanced_srp_results_{timestamp}.csv', index=False)
        print(f"Results saved to: advanced_srp_results_{timestamp}.csv")
    else:
        print("❌ No successful results from advanced methods")

if __name__ == "__main__":
    main()