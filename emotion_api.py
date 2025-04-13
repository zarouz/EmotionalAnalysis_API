# emotion_api.py
import os
import logging
from flask import Flask, request, jsonify

# Import your existing confidence analyzer logic
# Make sure confidence_analyzer.py is in the SAME directory
import confidence_analyzer

# --- Configuration ---
# Configure logging for the API service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get Flask port from environment variable or use default
API_PORT = int(os.environ.get("EMOTION_API_PORT", 5003)) # Use a different port, e.g., 5003


# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model ONCE on Startup ---
# Load the emotion model when the API server starts
# This prevents reloading the model for every request, improving performance.
try:
    logger.info("Attempting to load emotion model for API service...")
    confidence_analyzer.load_emotion_model()
    logger.info("Emotion model loaded successfully for API service.")
except Exception as e:
    logger.critical(f"CRITICAL ERROR: Could not load emotion model on API startup. API will likely fail. Error: {e}", exc_info=True)
    # Depending on your needs, you might want the app to exit if the model fails to load.
    # exit(1)


# --- API Endpoint Definition ---
@app.route('/analyze', methods=['POST'])
def analyze_audio_endpoint():
    """
    API endpoint to analyze emotion confidence from an audio file path.
    Expects JSON payload: {"audio_path": "/path/to/audio.wav"}
    Returns JSON response with analysis results.
    """
    logger.info("Received request for /analyze endpoint.")

    # --- Input Validation ---
    if not request.is_json:
        logger.warning("Request is not JSON.")
        return jsonify({"error": True, "message": "Request must be JSON"}), 400

    data = request.get_json()
    audio_path = data.get('audio_path')

    if not audio_path:
        logger.warning("Missing 'audio_path' in JSON payload.")
        return jsonify({"error": True, "message": "Missing 'audio_path' in JSON payload"}), 400

    logger.info(f"Received audio path: {audio_path}")

    # --- Security/Path Check ---
    # Basic check: Ensure the path exists *on the server running this API*.
    # WARNING: This assumes the main app and API service share a filesystem or the path is accessible.
    # If running in Docker or different machines, you'll need to handle file transfer
    # (e.g., upload the file data instead of sending the path).
    if not os.path.exists(audio_path):
         logger.error(f"Audio file not found at the provided path (on API server): {audio_path}")
         return jsonify({"error": True, "message": f"Audio file not found on server at path: {audio_path}"}), 404 # Not Found

    # --- Perform Analysis ---
    try:
        logger.info(f"Calling confidence_analyzer for path: {audio_path}")
        # Call your existing analysis function
        results = confidence_analyzer.analyze_confidence(audio_path)
        logger.info(f"Analysis complete for {audio_path}. Error status: {results.get('error')}")

        # Ensure the response is JSON serializable (should be if it's dicts/lists/primitives)
        return jsonify(results), 200

    except FileNotFoundError as fnf_err:
        # This might happen if analyze_confidence itself has issues finding sub-files etc.
        logger.error(f"FileNotFoundError during analysis for {audio_path}: {fnf_err}", exc_info=True)
        return jsonify({"error": True, "message": f"Internal analysis error: File not found - {fnf_err}"}), 500
    except RuntimeError as rt_err:
         # Catch specific runtime errors potentially from model loading/prediction if it wasn't caught on startup
         logger.error(f"RuntimeError during analysis for {audio_path}: {rt_err}", exc_info=True)
         return jsonify({"error": True, "message": f"Internal analysis error: Runtime Error - {rt_err}"}), 500
    except Exception as e:
        # Catch-all for other unexpected errors during analysis
        logger.error(f"Unexpected error during analysis for {audio_path}: {e}", exc_info=True)
        return jsonify({"error": True, "message": f"An unexpected internal server error occurred during analysis."}), 500


# --- Run the Flask App ---
if __name__ == '__main__':
    logger.info(f"Starting Emotion Analysis API server on port {API_PORT}...")
    # Use host='0.0.0.0' to make it accessible from other machines/containers on the network
    # Use debug=False for production/stable deployment
    app.run(host='0.0.0.0', port=API_PORT, debug=False)