# confidence_analyzer.py
import os
import numpy as np
import librosa
import tensorflow as tf
# tensorflow.keras layers might be needed for custom object scope if direct loading fails later
# from tensorflow.keras import layers
import warnings
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import tempfile
import soundfile as sf
import webrtcvad # Ensure this is installed: pip install webrtcvad-wheels
import logging

logger = logging.getLogger(__name__)

# Suppress common Librosa warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# --- Configuration ---
MODEL_PATH = "emotions_model/end_to_end_emotion_model.keras"

# Feature & Model parameters (MUST MATCH TRAINING and your inference script)
SR_TARGET_ENC_ATT = 16000
N_MFCC_ENC_ATT = 40
MAX_LEN_ENC_ATT = 300
FEATURE_DIM_ENC_ATT = N_MFCC_ENC_ATT + 7 + 12  # 59
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)

# Confidence weights (Adjust as needed)
CONFIDENCE_WEIGHTS = {
    'angry': 0.3, 'disgust': 0.5, 'fear': 0.1, 'happy': 0.9,
    'neutral': 0.8, 'sad': 0.2, 'surprise': 0.4
}

# --- Global Model Variable ---
loaded_emotion_model = None

# --- Model Loading Function (Simplified) ---
def load_emotion_model():
    """Loads the Keras emotion model directly using tf.keras.models.load_model."""
    global loaded_emotion_model
    if loaded_emotion_model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Emotion model file not found at '{MODEL_PATH}'")
            raise FileNotFoundError(f"Emotion model not found: {MODEL_PATH}")
        try:
            logger.info(f"Loading emotion model from '{MODEL_PATH}'...")
            # --- Direct Loading Call ---
            # Set compile=False if you don't need the optimizer state for inference
            # IMPORTANT: This environment (API Service) should NOT have tf-keras installed.
            loaded_emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            # --------------------------
            logger.info("Emotion model loaded successfully.")
            loaded_emotion_model.summary(print_fn=logger.info) # Log summary on first load
        except Exception as e:
            logger.error(f"Error loading Keras emotion model directly: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load emotion model directly: {e}") from e
    return loaded_emotion_model

# --- Feature Extraction (Use the one from your inference script) ---
def extract_features_encoder_attention(file_path, sr=SR_TARGET_ENC_ATT, n_mfcc=N_MFCC_ENC_ATT, max_len=MAX_LEN_ENC_ATT):
    """
    Extracts features matching the inference script.
    """
    try:
        audio, current_sr = librosa.load(file_path, sr=None, mono=True) # Load native SR
        if current_sr != sr:
             logger.info(f"Resampling audio from {current_sr} Hz to {sr} Hz.")
             audio = librosa.resample(y=audio, orig_sr=current_sr, target_sr=sr)
        if len(audio) < 100:
            logger.warning(f"Audio too short after potential resampling: {len(audio)} samples. Path: {file_path}")
            return None # Return None for very short audio
    except Exception as e:
        logger.error(f"Error loading/resampling audio {file_path}: {e}", exc_info=True)
        return None

    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T

        if mfccs.shape[0] < 2:
            logger.warning(f"Insufficient frames ({mfccs.shape[0]}) after feature extraction for {file_path}. Cannot normalize.")
            # Decide how to handle: return None or try padding zeros? Returning None is safer.
            return None

        # Normalize (handle potential zero std dev)
        mfccs_mean, mfccs_std = np.mean(mfccs, axis=0), np.std(mfccs, axis=0)
        contrast_mean, contrast_std = np.mean(contrast, axis=0), np.std(contrast, axis=0)
        chroma_mean, chroma_std = np.mean(chroma, axis=0), np.std(chroma, axis=0)

        # Use np.divide for safe division, replacing potential NaN/inf with 0
        mfccs = np.divide(mfccs - mfccs_mean, mfccs_std + 1e-10, out=np.zeros_like(mfccs), where=mfccs_std > 1e-10)
        contrast = np.divide(contrast - contrast_mean, contrast_std + 1e-10, out=np.zeros_like(contrast), where=contrast_std > 1e-10)
        chroma = np.divide(chroma - chroma_mean, chroma_std + 1e-10, out=np.zeros_like(chroma), where=chroma_std > 1e-10)

        # Combine features, ensuring no NaNs remain after division
        features = np.nan_to_num(np.hstack([mfccs, contrast, chroma]))

        # Pad/Truncate
        current_len = features.shape[0]
        if current_len == 0: # Double check length after normalization/stacking
             logger.warning(f"Zero frames after feature processing for {file_path}")
             return None
        if current_len < max_len:
            features = np.pad(features, ((0, max_len - current_len), (0, 0)), mode='constant')
        elif current_len > max_len:
            features = features[:max_len, :]

        # Final dimension check
        expected_shape = (max_len, FEATURE_DIM_ENC_ATT)
        if features.shape != expected_shape:
            logger.error(f"Feature dimension mismatch for {file_path}. Expected {expected_shape}, got {features.shape}.")
            # Attempt reshape if only feature dim is wrong? Risky. Best to return None.
            # Example recovery (use with caution):
            # if features.shape[0] == max_len and features.shape[1] < FEATURE_DIM_ENC_ATT:
            #    features = np.pad(features, ((0,0), (0, FEATURE_DIM_ENC_ATT - features.shape[1])), mode='constant')
            # elif features.shape[0] == max_len and features.shape[1] > FEATURE_DIM_ENC_ATT:
            #     features = features[:, :FEATURE_DIM_ENC_ATT]
            # else: return None # If lengths also mismatch
            return None
        return features.astype(np.float32)

    except Exception as e:
        logger.error(f"Error extracting features for {file_path}: {e}", exc_info=True)
        return None


# --- VAD Segmentation ---
def split_audio_by_speech(audio_path, sr=SR_TARGET_ENC_ATT, min_segment_length=3.0, vad_aggressiveness=2):
    """Splits audio into segments based on voice activity detection."""
    segments_data = []
    try:
        # Load audio with target sample rate
        audio, current_sr = librosa.load(audio_path, sr=None, mono=True)
        if current_sr != sr:
            logger.info(f"Resampling audio for VAD from {current_sr} Hz to {sr} Hz.")
            audio = librosa.resample(y=audio, orig_sr=current_sr, target_sr=sr)

        if len(audio) < sr * 0.1: # Check length *after* resampling
            logger.warning(f"Audio too short for VAD after resampling ({len(audio)} samples). Path: {audio_path}")
            return None, None # VAD cannot process very short audio

        # Convert to 16-bit PCM for VAD
        audio_int = np.int16(audio * 32767)

        # VAD setup (requires 10, 20, or 30 ms frames)
        vad = webrtcvad.Vad(vad_aggressiveness)
        frame_duration_ms = 30 # Use 30ms frames for compatibility
        frame_size = int(sr * frame_duration_ms / 1000)
        num_frames = len(audio_int) // frame_size

        is_speaking = False
        segment_start_frame = 0
        min_speech_frames = int((min_segment_length * 1000) / frame_duration_ms)
        padding_frames = int(0.1 * 1000 / frame_duration_ms) # 100ms padding

        for i in range(num_frames):
            start_sample = i * frame_size
            end_sample = start_sample + frame_size
            frame = audio_int[start_sample:end_sample]

            # Ensure frame has the correct number of bytes
            if len(frame) < frame_size:
                # Pad the last frame if necessary
                frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')

            try:
                # VAD expects bytes
                frame_is_speech = vad.is_speech(frame.tobytes(), sr)
            except Exception as vad_err:
                 # Handle potential errors like wrong frame length/rate for VAD
                 logger.warning(f"VAD error on frame {i}: {vad_err}. Treating as non-speech.")
                 frame_is_speech = False

            if not is_speaking and frame_is_speech:
                is_speaking = True
                segment_start_frame = max(0, i - padding_frames) # Start segment slightly before speech begins
            elif is_speaking and not frame_is_speech:
                # End of speech segment detected
                if (i - segment_start_frame) >= min_speech_frames:
                    # Segment is long enough, store it
                    segment_end_frame = min(num_frames, i + padding_frames) # End segment slightly after speech ends
                    start_sample_final = segment_start_frame * frame_size
                    end_sample_final = segment_end_frame * frame_size
                    segments_data.append(audio[start_sample_final:end_sample_final])
                is_speaking = False # Reset speaking flag

        # Handle case where speech continues to the end of the file
        if is_speaking and (num_frames - segment_start_frame) >= min_speech_frames:
             segment_end_frame = num_frames # Go to the very end
             start_sample_final = segment_start_frame * frame_size
             end_sample_final = segment_end_frame * frame_size # Should be len(audio) effectively
             segments_data.append(audio[start_sample_final:end_sample_final])

        # Check results
        if not segments_data:
             total_duration = len(audio) / sr
             if total_duration >= min_segment_length:
                  logger.warning(f"VAD found no speech segments in {os.path.basename(audio_path)} ({total_duration:.1f}s long), returning full audio as one segment.")
                  return [audio], sr # Return full audio if long enough but no speech detected
             else:
                  logger.warning(f"VAD found no speech segments and audio too short ({total_duration:.1f}s) in {os.path.basename(audio_path)}.")
                  return None, None
        else:
             logger.info(f"VAD split {os.path.basename(audio_path)} into {len(segments_data)} speech segments.")
             return segments_data, sr

    except Exception as e:
        logger.error(f"Error in VAD processing for {audio_path}: {e}", exc_info=True)
        return None, None

# --- Weighted Aggregation ---
def aggregate_results(segment_results):
    """Aggregates emotion probabilities from segments using duration and confidence weighting."""
    if not segment_results:
        logger.warning("aggregate_results called with no segment results.")
        return None # Return None if no segments to aggregate

    valid_segments = [s for s in segment_results if 'probs' in s and s['probs'] is not None and s['probs'].shape == (NUM_CLASSES,)]
    if not valid_segments:
        logger.warning("No valid segments found for aggregation (missing or invalid 'probs').")
        return None

    # Initialize weighted probabilities array
    weighted_probs = np.zeros(NUM_CLASSES, dtype=np.float64) # Use float64 for accumulation precision
    total_weight = 0.0

    for seg in valid_segments:
        confidence = seg.get('confidence', 0.0) # Default confidence to 0 if missing
        duration = seg.get('duration', 0.0)    # Default duration to 0 if missing
        probs = seg.get('probs')

        # Basic validation
        if not isinstance(confidence, (int, float)) or not isinstance(duration, (int, float)):
            logger.warning(f"Skipping segment with invalid confidence/duration types: {seg}")
            continue
        if duration < 0.1: # Skip very short segments in aggregation as well
             logger.debug(f"Skipping very short segment (duration {duration:.2f}s) in aggregation.")
             continue

        # Calculate weight (adjust formula as needed)
        # Example: Weight by duration, slightly boosted by prediction confidence
        weight = duration * (1.0 + (confidence / 100.0) * 0.5) # Give confidence less impact than duration
        # Alternative: Simple duration weighting: weight = duration

        weighted_probs += probs * weight
        total_weight += weight
        logger.debug(f"Segment {seg.get('segment', 'N/A')}: Dur={duration:.2f}s, Conf={confidence:.1f}%, Weight={weight:.3f}")

    # Normalize the weighted probabilities
    if total_weight <= 1e-9: # Use a small epsilon to avoid division by zero
        logger.warning("Total weight for aggregation is near zero. Falling back to simple average.")
        # Fallback: Calculate simple average if total weight is negligible
        valid_probs_list = [seg['probs'] for seg in valid_segments]
        if not valid_probs_list: return None # Should not happen if valid_segments exist, but safety check
        final_probs = np.mean(valid_probs_list, axis=0)
    else:
        final_probs = weighted_probs / total_weight

    # Ensure probabilities sum reasonably close to 1 (optional check)
    if not np.isclose(np.sum(final_probs), 1.0, atol=1e-5):
        logger.warning(f"Aggregated probabilities do not sum close to 1 ({np.sum(final_probs):.4f}). Re-normalizing.")
        final_probs /= np.sum(final_probs) # Re-normalize if needed

    return final_probs.astype(np.float32) # Return as float32


# --- Main VAD-based Processor ---
def process_long_audio_vad(audio_path, model, classes, feature_params):
    """Processes long audio using VAD and the pre-loaded model."""
    segments, sr = split_audio_by_speech(
        audio_path, sr=feature_params['sr'],
        min_segment_length=1.0, # Consider reducing min length slightly if needed
        vad_aggressiveness=2 # 0=least aggressive, 3=most aggressive
    )
    if segments is None:
        logger.error(f"VAD processing failed for {os.path.basename(audio_path)}")
        return None, [], True # Indicate error state
    if not segments:
        logger.warning(f"No valid speech segments found by VAD for {os.path.basename(audio_path)}")
        # If VAD returns empty list but audio might be okay, maybe process whole file?
        # Or return error as originally intended. Returning error state for now.
        return None, [], True

    segment_results = []
    logger.info(f"Processing {len(segments)} speech segments from {os.path.basename(audio_path)}...")

    for i, segment_audio in enumerate(segments):
        duration = len(segment_audio) / sr
        logger.debug(f"Processing segment {i+1}/{len(segments)}, Duration: {duration:.2f}s")

        # Skip segments that are too short even if VAD returned them
        if duration < 0.5: # Minimum duration for reliable feature extraction/prediction
            logger.info(f"Skipping segment {i+1} because it's too short ({duration:.2f}s).")
            continue

        temp_filename = None
        features = None
        segment_result_item = {'segment': i + 1, 'duration': duration, 'error': True, 'message': 'Processing not completed'}

        try:
            # Create a temporary WAV file for the segment
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                temp_filename = tmpfile.name
            sf.write(temp_filename, segment_audio, sr)
            logger.debug(f"Segment {i+1} saved to temp file: {temp_filename}")

            # Extract features from the temporary file
            features = extract_features_encoder_attention(
                temp_filename, sr=feature_params['sr'],
                n_mfcc=feature_params['n_mfcc'], max_len=feature_params['max_len']
            )
        except Exception as proc_err:
            logger.error(f"Error preparing/extracting features for segment {i+1}: {proc_err}", exc_info=True)
            segment_result_item['message'] = f"Feature extraction error: {proc_err}"
            # Continue to finally block for cleanup
        finally:
            # Ensure temporary file is always removed
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                    logger.debug(f"Removed temp file: {temp_filename}")
                except OSError as e:
                    logger.warning(f"Could not remove temp file {temp_filename}: {e}")

        # If feature extraction failed, store error and continue to next segment
        if features is None:
            logger.warning(f"Segment {i+1} feature extraction failed or produced no features.")
            if 'Feature extraction error' not in segment_result_item['message']: # Update message if not already set
                 segment_result_item['message'] = "Feature extraction failed or yielded no features"
            segment_results.append(segment_result_item)
            continue

        # Prepare features for the model prediction
        features_batch = np.expand_dims(features, axis=0)

        # Predict using the globally loaded model
        try:
            logger.debug(f"Predicting for segment {i+1}...")
            # --- Uses the model loaded by load_emotion_model ---
            probs = model.predict(features_batch, verbose=0)[0] # Get the first (only) item in the batch prediction
            # ---------------------------------------------------

            if probs is None or probs.shape != (len(classes),):
                logger.warning(f"Segment {i+1} prediction yielded invalid probabilities shape. Got: {probs.shape if probs is not None else 'None'}")
                segment_result_item['message'] = "Model prediction yielded invalid shape"
                segment_results.append(segment_result_item)
                continue

            pred_index = np.argmax(probs)
            pred_confidence = probs[pred_index] * 100

            # Store successful segment result
            segment_results.append({
                'segment': i + 1,
                'emotion': classes[pred_index],
                'confidence': pred_confidence, # This is likely float32
                'probs': probs,               # This is likely float32 array
                'duration': duration,         # Standard float
                'error': False,
                'message': 'Success'
            })
            logger.debug(f"Segment {i+1}: Predicted '{classes[pred_index]}' ({pred_confidence:.1f}%), Duration: {duration:.2f}s")

        except Exception as pred_err:
            logger.error(f"Error predicting segment {i+1}: {pred_err}", exc_info=True)
            segment_result_item['message'] = f"Model prediction error: {pred_err}"
            segment_results.append(segment_result_item)
            continue # Continue to the next segment

    # After processing all segments
    if not segment_results:
        logger.error(f"Processing failed or yielded no valid results for any segment in {os.path.basename(audio_path)}")
        return None, [], True # Error state if no segments could be processed at all

    # Aggregate results from successfully processed segments
    final_probs = aggregate_results([s for s in segment_results if not s['error']]) # Only aggregate successful ones
    if final_probs is None:
        logger.error(f"Aggregation failed for {os.path.basename(audio_path)}, possibly no valid segments succeeded.")
        # Return segment details, but flag overall error
        return None, segment_results, True

    # Aggregation successful
    return final_probs, segment_results, False # Return aggregated probs, details, and success flag


# --- Confidence Scoring & Rating ---
def calculate_confidence_score(emotion_probs, emotion_classes, confidence_weights):
    """Calculates a weighted confidence score based on emotion probabilities."""
    if emotion_probs is None or len(emotion_probs) != len(emotion_classes):
         logger.error("Invalid input for calculate_confidence_score.")
         return 0.0 # Return a default score

    weighted_sum = 0.0
    total_weight_used = 0.0 # Track sum of weights used for normalization if needed
    matched_classes = 0

    for i, emotion in enumerate(emotion_classes):
        weight = confidence_weights.get(emotion)
        if weight is not None:
            weighted_sum += emotion_probs[i] * weight
            total_weight_used += weight # Sum the weights corresponding to matched emotions
            matched_classes += 1
        else:
             logger.warning(f"Emotion '{emotion}' not found in CONFIDENCE_WEIGHTS config.")

    if matched_classes == 0:
        logger.error("No emotion classes matched weights configuration!")
        return 0.0
    if total_weight_used <= 1e-9:
         logger.warning("Sum of weights used is near zero, cannot calculate meaningful score.")
         return 0.0

    # Normalize the weighted sum? Optional - depends if you want score relative to max possible weighted score
    # max_possible_score = total_weight_used # If weights aren't normalized, this is the max
    # confidence_score = (weighted_sum / max_possible_score) * 100 if max_possible_score > 0 else 0

    # Simpler: Just scale the weighted sum
    # Adjust scaling factor if needed based on typical weight values
    confidence_score = weighted_sum * 100

    # Clamp the score between 0 and 100
    confidence_score = min(max(confidence_score, 0.0), 100.0)

    return confidence_score # This should be a standard Python float

def get_confidence_rating(score):
    """Assigns a qualitative rating based on the numerical confidence score."""
    if score is None:
        return "N/A"
    # Ensure score is float for comparison
    try:
        score = float(score)
    except (ValueError, TypeError):
         logger.warning(f"Invalid score type '{type(score)}' for get_confidence_rating.")
         return "Invalid Score"

    if score >= 80: return "Very High"
    elif score >= 65: return "High"
    elif score >= 45: return "Moderate"
    elif score >= 25: return "Low"
    else: return "Very Low"

# --- Main Analysis Function ---
def analyze_confidence(audio_path):
    """
    Performs full confidence analysis on an audio file.
    Returns a dictionary with results, including converting numpy types for JSON compatibility.
    """
    # Default result structure
    results = {
        'score': None, 'rating': "N/A", 'primary_emotion': "N/A",
        'emotion_confidence': None, 'segment_results': [],
        'all_probs': None, 'error': True, 'message': 'Analysis not started'
    }
    try:
        # --- Load model (uses global) ---
        model = load_emotion_model()
        if model is None:
            # This case should be caught by load_emotion_model raising an error,
            # but double-check for robustness.
            raise RuntimeError("Emotion model is None after load_emotion_model call.")

    except Exception as load_err:
        logger.error(f"Failed to load model for analysis: {load_err}", exc_info=True)
        results['message'] = f"Model loading failed: {load_err}"
        # No need for JSON conversion here as it's just default strings/None
        return results # Return default error state

    # Define feature parameters
    feature_params = {'sr': SR_TARGET_ENC_ATT, 'n_mfcc': N_MFCC_ENC_ATT, 'max_len': MAX_LEN_ENC_ATT}

    # Process audio using VAD
    agg_probs, segment_details, error_flag = process_long_audio_vad(
        audio_path, model, EMOTION_CLASSES, feature_params
    )
    # Update segment results regardless of overall success/failure
    results['segment_results'] = segment_details

    if error_flag or agg_probs is None:
        logger.error(f"Analysis failed during processing or aggregation for {os.path.basename(audio_path)}")
        # Update message if not already set by specific errors
        if results['message'] == 'Analysis not started':
             results['message'] = "VAD processing or result aggregation failed."
        # Keep error=True
    else:
        # Analysis successful, calculate final metrics
        predicted_index = np.argmax(agg_probs)
        primary_emotion = EMOTION_CLASSES[predicted_index]
        # Ensure this is converted later
        emotion_confidence_percent = agg_probs[predicted_index] * 100
        # Ensure this is converted later
        overall_score = calculate_confidence_score(agg_probs, EMOTION_CLASSES, CONFIDENCE_WEIGHTS)
        rating = get_confidence_rating(overall_score)

        results.update({
            'score': overall_score, # Potential numpy float
            'rating': rating,       # String
            'primary_emotion': primary_emotion, # String
            'emotion_confidence': emotion_confidence_percent, # Potential numpy float
            'all_probs': agg_probs.tolist(), # .tolist() converts numpy array to Python list of floats
            'error': False,
            'message': 'Analysis successful'
        })
        logger.info(f"Confidence analysis complete for {os.path.basename(audio_path)}: Score={overall_score:.1f}, Rating={rating}")

    # --- JSON Conversion ---
    # Convert numpy types within the results dict before returning
    logger.debug(f"Results before JSON conversion: {results}")
    cleaned_results = {}
    for key, value in results.items():
        if isinstance(value, np.generic):
            # Convert numpy numeric types (float32, int64 etc.) to standard Python types
            cleaned_results[key] = value.item()
        elif isinstance(value, (list, tuple)):
            # Handle lists (like segment_results or all_probs if .tolist() wasn't used)
            cleaned_list = []
            for item in value:
                if isinstance(item, dict): # Handle list of dicts (like segment_results)
                    cleaned_dict_item = {}
                    for k, v in item.items():
                        if isinstance(v, np.generic):
                            cleaned_dict_item[k] = v.item()
                        elif isinstance(v, np.ndarray): # Handle numpy arrays within dicts
                            cleaned_dict_item[k] = v.tolist()
                        else:
                            cleaned_dict_item[k] = v
                    cleaned_list.append(cleaned_dict_item)
                elif isinstance(item, np.generic): # Handle list of numpy numbers
                    cleaned_list.append(item.item())
                elif isinstance(item, np.ndarray): # Handle list of numpy arrays
                    cleaned_list.append(item.tolist())
                else:
                    cleaned_list.append(item) # Keep non-numpy items as is
            cleaned_results[key] = cleaned_list
        elif isinstance(value, dict):
            # Handle simple dictionaries (if any other than segment items)
            cleaned_dict = {}
            for k, v in value.items():
                 if isinstance(v, np.generic):
                      cleaned_dict[k] = v.item()
                 elif isinstance(v, np.ndarray): # Handle numpy arrays within dicts
                      cleaned_dict[k] = v.tolist()
                 else:
                      cleaned_dict[k] = v
            cleaned_results[key] = cleaned_dict
        elif isinstance(value, np.ndarray): # Handle top-level numpy arrays (e.g., if all_probs wasn't tolist()'ed)
             cleaned_results[key] = value.tolist()
        else:
            # Keep standard Python types (str, int, float, bool, None, etc.) as they are
            cleaned_results[key] = value

    logger.debug(f"Results after JSON conversion: {cleaned_results}")
    return cleaned_results # Return the JSON-safe dictionary


# --- Visualization Function (Keep or remove if not used by API) ---
def visualize_confidence_analysis(agg_probs, score, rating, classes=EMOTION_CLASSES):
    # ... (Keep the function code from your provided confidence script if needed) ...
    # This likely won't be called directly by the API, but keep it if you might use it elsewhere.
    pass

# --- Example Usage (If run directly - useful for testing the API's core logic) ---
if __name__ == "__main__":
    # Setup basic logging for direct run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running confidence_analyzer.py directly for testing...")

    # --- IMPORTANT: Create a dummy model file or ensure the real one exists ---
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'. Cannot run test.")
        # Optionally create a dummy file structure for basic tests without loading
        # os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # with open(MODEL_PATH, 'w') as f: f.write("dummy model file")
        exit()

    # --- Create a dummy audio file for testing ---
    SAMPLE_RATE_TEST = 16000
    DURATION_TEST = 5 # seconds
    FREQUENCY_TEST = 440 # Hz (A4 note)
    test_audio_filename = "test_confidence_audio.wav"

    try:
        t = np.linspace(0., DURATION_TEST, int(SAMPLE_RATE_TEST * DURATION_TEST), endpoint=False)
        # Generate a simple sine wave with some noise
        audio_signal = 0.5 * np.sin(2. * np.pi * FREQUENCY_TEST * t) + 0.05 * np.random.randn(len(t))
        # Ensure it's float32 for soundfile
        audio_signal = audio_signal.astype(np.float32)
        sf.write(test_audio_filename, audio_signal, SAMPLE_RATE_TEST)
        print(f"Created dummy test audio file: {test_audio_filename}")

        # --- Test the analysis function ---
        print("\n--- Testing analyze_confidence ---")
        analysis_result = analyze_confidence(test_audio_filename)
        print("\nAnalysis Result:")
        import json
        print(json.dumps(analysis_result, indent=2)) # Pretty print the JSON-ready result

    except FileNotFoundError as e:
         print(f"Error during test setup (Model not found?): {e}")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
        logger.error("Error during direct test run", exc_info=True)
    finally:
        # Clean up the dummy audio file
        if os.path.exists(test_audio_filename):
            try:
                os.remove(test_audio_filename)
                print(f"Cleaned up {test_audio_filename}")
            except OSError as e:
                print(f"Could not remove test file {test_audio_filename}: {e}")