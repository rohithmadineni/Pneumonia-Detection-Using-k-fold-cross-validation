from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import uuid  # Added import for uuid
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/MobileNetV2_Final.h5'  # Path to your saved model

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model for pneumonia detection
# At the beginning of your app.py
try:
    pneumonia_model = load_model(MODEL_PATH)
    print(f"Pneumonia model loaded successfully! Input shape: {pneumonia_model.input_shape}")
    print(f"Model output shape: {pneumonia_model.output_shape}")
except Exception as e:
    print(f"Error loading pneumonia model: {str(e)}")
    pneumonia_model = None

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to check if an image is likely a chest X-ray
def is_chest_xray(img_array):
    """
    Validates if an image is a chest X-ray using multiple techniques.
    Returns True if the image appears to be a chest X-ray, False otherwise.
    """
    try:
        # Work with the first image if batched
        single_img = img_array[0]
        
        # 1. Grayscale analysis - X-rays are essentially grayscale
        if single_img.shape[-1] == 3:  # If RGB
            # Calculate mean of each channel
            r_mean, g_mean, b_mean = np.mean(single_img[:,:,0]), np.mean(single_img[:,:,1]), np.mean(single_img[:,:,2])
            
            # Calculate standard deviation between channel means
            channel_means = [r_mean, g_mean, b_mean]
            channel_std = np.std(channel_means)
            
            # Calculate maximum difference between any two channels
            max_diff = max([abs(channel_means[i] - channel_means[j]) 
                           for i in range(3) for j in range(i+1, 3)])
            
            print(f"Channel means: R={r_mean:.4f}, G={g_mean:.4f}, B={b_mean:.4f}")
            print(f"Channel std: {channel_std:.4f}, Max diff: {max_diff:.4f}")
            
            # True X-rays converted to RGB usually have very similar channel values
            if channel_std > 0.05 or max_diff > 0.1:
                print("Failed grayscale check: High color variation between channels")
                return False
        
        # 2. Convert to grayscale for further analysis
        if single_img.shape[-1] == 3:
            gray_img = np.mean(single_img, axis=-1)
        else:
            gray_img = single_img.squeeze()
        
        # 3. Brightness and contrast analysis
        img_mean = np.mean(gray_img)
        img_std = np.std(gray_img)
        
        print(f"Image brightness (mean): {img_mean:.4f}")
        print(f"Image contrast (std): {img_std:.4f}")
        
        # X-rays typically have moderate brightness and good contrast
        if img_mean < 0.2 or img_mean > 0.8:
            print("Failed brightness check: Image too dark or too bright")
            return False
            
        if img_std < 0.1:
            print("Failed contrast check: Image has insufficient contrast")
            return False
        
        # 4. Histogram analysis for tissue differentiation
        hist, bins = np.histogram(gray_img, bins=50, range=(0, 1))
        hist_normalized = hist / np.sum(hist)
        
        # Calculate peaks in histogram
        peak_threshold = 0.02
        peaks = []
        for i in range(1, len(hist_normalized)-1):
            if (hist_normalized[i] > hist_normalized[i-1] and 
                hist_normalized[i] > hist_normalized[i+1] and
                hist_normalized[i] > peak_threshold):
                peaks.append((bins[i], hist_normalized[i]))
        
        print(f"Number of significant peaks in histogram: {len(peaks)}")
        
        # X-rays typically have 2-5 distinct density regions
        if len(peaks) < 2 or len(peaks) > 8:
            print("Failed histogram check: Unusual density distribution")
            # This is a weak signal, so we'll just warn rather than reject
            # return False
        
        # 5. Edge detection to look for anatomical structures
        # Compute simple edge map using Sobel-like gradients
        gx = np.abs(np.gradient(gray_img, axis=0))
        gy = np.abs(np.gradient(gray_img, axis=1))
        edges = np.sqrt(gx**2 + gy**2)
        
        # Analyze edge statistics
        edge_threshold = 0.05
        edge_percentage = np.mean(edges > edge_threshold)
        edge_mean = np.mean(edges)
        
        print(f"Edge percentage: {edge_percentage:.4f}")
        print(f"Edge mean strength: {edge_mean:.4f}")
        
        # X-rays have a moderate amount of edges from anatomical structures
        if edge_percentage < 0.01 or edge_percentage > 0.3:
            print("Failed edge check: Unusual edge distribution")
            return False
            
        # 6. Check for abnormal patterns
        # Detect if image has very regular patterns like screenshots or diagrams
        # We'll use a simple check based on row/column averages
        row_means = np.mean(gray_img, axis=1)
        col_means = np.mean(gray_img, axis=0)
        
        # Calculate variation in row/column means
        row_var = np.var(row_means)
        col_var = np.var(col_means)
        
        print(f"Row variance: {row_var:.6f}, Column variance: {col_var:.6f}")
        
        # Extremely low variance might indicate artificially generated images
        if row_var < 0.0001 or col_var < 0.0001:
            print("Failed pattern check: Too regular patterns detected")
            return False
        
        # 7. Spatial consistency check
        # In real X-rays, the center is usually more "filled" than edges
        h, w = gray_img.shape
        center_region = gray_img[h//4:3*h//4, w//4:3*w//4]
        edge_region = np.concatenate([
            gray_img[:h//4, :],  # top
            gray_img[3*h//4:, :],  # bottom
            gray_img[h//4:3*h//4, :w//4],  # left
            gray_img[h//4:3*h//4, 3*w//4:]  # right
        ])
        
        center_mean = np.mean(center_region)
        edge_mean = np.mean(edge_region)
        
        print(f"Center region mean: {center_mean:.4f}, Edge region mean: {edge_mean:.4f}")
        
        # In most X-rays, the difference shouldn't be extreme
        if abs(center_mean - edge_mean) > 0.5:
            print("Failed spatial consistency check: Unusual center-edge difference")
            return False
        
        # If all checks pass, it's likely a chest X-ray
        print("Image passed all X-ray validation checks")
        return True
        
    except Exception as e:
        print(f"Error in X-ray validation: {str(e)}")
        # Fall back to a basic grayscale check
        try:
            # Simple check - is it mostly grayscale?
            if single_img.shape[-1] == 3:
                r, g, b = single_img[:,:,0], single_img[:,:,1], single_img[:,:,2]
                diff_rg = np.mean(np.abs(r - g))
                diff_rb = np.mean(np.abs(r - b))
                diff_gb = np.mean(np.abs(g - b))
                
                avg_diff = (diff_rg + diff_rb + diff_gb) / 3
                print(f"Average channel difference (fallback): {avg_diff:.4f}")
                
                # If channels are very similar, it's likely grayscale
                return avg_diff < 0.05
            return True
        except Exception as e:
            print(f"Error in fallback validation: {str(e)}")
            # If everything fails, err on the side of accepting the image
            return True

def preprocess_image(img):
    """
    Preprocess image for model input with added error handling
    """
    try:
        # Resize to the input size expected by your model
        target_size = (224, 224)  # Common size, adjust to your model's input size
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        
        # Normalize values to [0,1] range
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Print debug info
        print(f"Preprocessed image shape: {img_array.shape}")
        print(f"Value range: {np.min(img_array)} to {np.max(img_array)}")
        
        return img_array
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            # Create a unique filename
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            # Read and process the image
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # Save the file
            img.save(file_path)
            
            # Preprocess image
            img_array = preprocess_image(img)
            
            # Debug information
            print(f"Image shape after preprocessing: {img_array.shape}")
            print(f"Image min/max values: {np.min(img_array)}/{np.max(img_array)}")
            
            # Step 1: Check if the image is a chest X-ray
            try:
                is_valid_xray = is_chest_xray(img_array)
                print(f"Image validation result: {is_valid_xray}")
            except Exception as e:
                print(f"Error in X-ray validation: {str(e)}")
                return jsonify({
                    'error': f"Error validating image: {str(e)}",
                    'file_path': file_path
                })
            
            if not is_valid_xray:
                return jsonify({
                    'error': 'The uploaded image does not appear to be a chest X-ray.',
                    'file_path': file_path
                })
            
            # Step 2: Make prediction if model is loaded
            if pneumonia_model is not None:
                try:
                    prediction = pneumonia_model.predict(img_array)
                    print(f"Prediction shape: {prediction.shape}, value: {prediction}")
                    probability = float(prediction[0][0])  # Adjust based on your model's output format
                    is_pneumonia = probability > 0.5  # Adjust threshold as needed
                except Exception as e:
                    print(f"Error in model prediction: {str(e)}")
                    return jsonify({
                        'error': f"Error making prediction: {str(e)}",
                        'file_path': file_path
                    })
                
                # Comment out database logic for now
                # We'll implement database functionality later
                """
                if 'user_id' in session:
                    try:
                        conn = get_db_connection()
                        conn.execute(
                            'INSERT INTO predictions (user_id, image_path, is_pneumonia, probability) VALUES (?, ?, ?, ?)',
                            (session['user_id'], file_path, is_pneumonia, probability)
                        )
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        print(f"Error saving to database: {str(e)}")
                        # Continue even if database save fails
                """
                
                result = {
                    'is_pneumonia': is_pneumonia,
                    'probability': probability,
                    'file_path': file_path
                }
                return jsonify(result)
            else:
                return jsonify({'error': 'Model not loaded', 'file_path': file_path})
        
        return jsonify({'error': 'Invalid file format. Please upload a PNG or JPEG image.'})
    except Exception as e:
        import traceback
        print(f"Unhandled exception in predict route: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f"Processing error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)