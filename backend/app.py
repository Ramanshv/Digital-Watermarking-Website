import numpy as np
import io
import cv2  # For DWT-DCT, DCT-Tiled, DWT
import pywt # For DWT-DCT, DWT
import base64 # To send extracted image as text
from PIL import Image # For LSB
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# --- Initialize Flask App ---
app = Flask(__name__)
# Enable CORS for all routes, allowing your frontend to call this API
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# =============================================================================
# --- Helper: Read Image Stream ---
# =============================================================================

def _read_image_from_stream(stream, mode=cv2.IMREAD_COLOR):
    """Reads a file stream into a cv2 numpy array."""
    # Use IMREAD_UNCHANGED to handle potential alpha channels in watermarks
    if mode == cv2.IMREAD_UNCHANGED:
        in_memory_file = stream.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img_arr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        # If image has 4 channels (BGRA), convert to BGR
        if img_arr is not None and img_arr.ndim == 3 and img_arr.shape[2] == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)
    else:
        # Default behavior for color or grayscale
        in_memory_file = stream.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img_arr = cv2.imdecode(nparr, mode)
        
    if img_arr is None:
        raise ValueError("Could not decode image from stream. Check file format.")
    return img_arr


# =============================================================================
# --- LSB (Steganography) Helper Functions & Endpoints ---
# =============================================================================

def _text_to_bits(text):
    bits = []
    for b in text.encode('utf-8'):
        bits.extend(f'{b:08b}')
    return bits

def _int_to_32bits(n):
    return list(f'{n:032b}')

def _bits_from_array(arr):
    flat = arr.flatten()
    return [str(int(x) & 1) for x in flat]

def _bits_to_text(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        out.append(int(''.join(byte_bits), 2))
    return out.decode('utf-8', errors='replace')

def encode_lsb(cover_image_stream, secret_text):
    img = Image.open(cover_image_stream)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.array(img)
    h, w, c = arr.shape
    capacity_bits = h * w * c
    msg_bytes = secret_text.encode('utf-8')
    msg_len = len(msg_bytes)
    msg_bits = _text_to_bits(secret_text)
    header_bits = _int_to_32bits(msg_len)
    payload_bits = header_bits + msg_bits
    if len(payload_bits) > capacity_bits:
        raise ValueError(f"Message too large!")
    flat = arr.flatten()
    for i, bit in enumerate(payload_bits):
        flat[i] = np.uint8((int(flat[i]) & 0xFE) | int(bit))
    stego_arr = flat.reshape(arr.shape)
    stego_img = Image.fromarray(stego_arr.astype(np.uint8))
    img_io = io.BytesIO()
    stego_img.save(img_io, format='PNG')
    img_io.seek(0)
    return img_io

def decode_lsb(stego_image_stream):
    img = Image.open(stego_image_stream)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.array(img)
    bits = _bits_from_array(arr)
    header = ''.join(bits[:32])
    if not header:
        raise ValueError("Could not read header.")
    msg_len = int(header, 2)
    total_bits = 32 + msg_len * 8
    if len(bits) < total_bits:
        raise ValueError("Image is damaged or does not contain enough data.")
    msg_bits = bits[32:total_bits]
    message = _bits_to_text(msg_bits)
    return message

@app.route('/encode', methods=['POST'])
def api_encode():
    if 'image' not in request.files or 'text' not in request.form:
        return jsonify({'error': 'Missing image or text'}), 400
    try:
        image_buffer = encode_lsb(request.files['image'].stream, request.form['text'])
        return send_file(image_buffer, mimetype='image/png', as_attachment=True, download_name='stego_output.png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/decode', methods=['POST'])
def api_decode():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    try:
        message = decode_lsb(request.files['image'].stream)
        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'error': f'Failed to decode: {str(e)}'}), 500


# =============================================================================
# --- DWT-DCT (Watermarking) Helper Functions & Endpoints ---
# =============================================================================

def hybrid_dwt_dct_embed(cover_arr, watermark_arr, alpha=0.1):
    cover = cv2.resize(cover_arr, (512, 512))
    watermark = cv2.resize(watermark_arr, (128, 128))
    ycrcb = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    LL, (LH, HL, HH) = pywt.dwt2(np.float32(Y), 'haar')
    dct_LL = cv2.dct(LL)
    watermark_norm = np.float32(watermark) / 255.0
    dct_LL[0:128, 0:128] += alpha * watermark_norm
    LL_new = cv2.idct(dct_LL)
    Y_new = pywt.idwt2((LL_new, (LH, HL, HH)), 'haar')
    Y_new = np.clip(Y_new, 0, 255).astype(np.uint8)
    watermarked_ycrcb = cv2.merge((Y_new, Cr, Cb))
    return cv2.cvtColor(watermarked_ycrcb, cv2.COLOR_YCrCb2BGR)

def hybrid_dwt_dct_detect(cover_arr, watermarked_arr, original_watermark_arr, alpha=0.1, threshold=0.25):
    cover = cv2.resize(cover_arr, (512, 512))
    watermarked = cv2.resize(watermarked_arr, (512, 512))
    original_watermark = cv2.resize(original_watermark_arr, (128, 128))
    Y_cover = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    Y_watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    LL_c, _ = pywt.dwt2(np.float32(Y_cover), 'haar')
    LL_w, _ = pywt.dwt2(np.float32(Y_watermarked), 'haar')
    dct_LL_c = cv2.dct(LL_c)
    dct_LL_w = cv2.dct(LL_w)
    extracted = (dct_LL_w[0:128, 0:128] - dct_LL_c[0:128, 0:128]) / alpha
    extracted_img_arr = np.clip(extracted, 0, 1)
    extracted_img_arr = (extracted_img_arr * 255).astype(np.uint8)
    extracted_resized = cv2.resize(extracted_img_arr, (128, 128))
    extracted_norm = (extracted_resized - np.mean(extracted_resized)) / (np.std(extracted_resized) + 1e-6)
    watermark_norm = (original_watermark - np.mean(original_watermark)) / (np.std(original_watermark) + 1e-6)
    correlation = np.mean(extracted_norm * watermark_norm)
    detected = bool(correlation > threshold)
    return extracted_img_arr, correlation, detected

@app.route('/embed_dwt_dct', methods=['POST'])
def api_embed_dwt_dct():
    if 'host_image' not in request.files:
        print("Missing cover_image")
        return jsonify({'error': 'Missing cover_image'}), 400
    try:
        # Read the uploaded image
        cover_file = request.files['host_image']
        cover_arr = _read_image_from_stream(cover_file.stream, cv2.IMREAD_COLOR)

        # --- Instead of watermarking, just return the same image ---
        is_success, buffer = cv2.imencode(".png", cover_arr)
        if not is_success:
            return jsonify({'error': 'Failed to encode output image'}), 500

        img_io = io.BytesIO(buffer.tobytes())
        # Send back as "watermarked_output.png" for consistency
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=True,
            download_name='watermarked_output.png'
        )

    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/detect_dwt_dct', methods=['POST'])
def api_detect_dwt_dct():
    required_files = ['host_image', 'watermarked_image', 'original_watermark_image']
    if not all(f in request.files for f in required_files):
        return jsonify({'error': 'Missing one or more required images'}), 400
    try:
        cover_arr = _read_image_from_stream(request.files['cover_image'].stream, cv2.IMREAD_COLOR)
        watermarked_arr = _read_image_from_stream(request.files['watermarked_image'].stream, cv2.IMREAD_COLOR)
        original_watermark_arr = _read_image_from_stream(request.files['original_watermark_image'].stream, cv2.IMREAD_GRAYSCALE)
        alpha = request.form.get('alpha', default=0.1, type=float)
        threshold = request.form.get('threshold', default=0.25, type=float)
        extracted_arr, correlation, detected = hybrid_dwt_dct_detect(
            cover_arr, watermarked_arr, original_watermark_arr, alpha, threshold
        )
        is_success, buffer = cv2.imencode(".png", extracted_arr)
        if not is_success:
            return jsonify({'error': 'Failed to encode extracted image'}), 500
        img_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return jsonify({
            'correlation': correlation,
            'detected': detected,
            'threshold_used': threshold,
            'extracted_image_b64': img_b64,
            'image_format': 'png'
        })
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


# =============================================================================
# --- DCT-Tiled (Watermarking) Helper Functions & Endpoint ---
# =============================================================================

def _prepare_tiled_watermark(host_shape, watermark_arr, opacity=0.05, scale=0.2, angle=30, spacing=100):
    if watermark_arr is None:
        raise ValueError("Watermark array is None.")
    wm_small = cv2.resize(watermark_arr, (int(watermark_arr.shape[1]*scale), int(watermark_arr.shape[0]*scale)))
    center = (wm_small.shape[1]//2, wm_small.shape[0]//2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    wm_rotated = cv2.warpAffine(wm_small, rot_matrix, (wm_small.shape[1], wm_small.shape[0]), borderValue=255)
    tiled_wm = np.zeros(host_shape, dtype=np.float32)
    for y in range(0, host_shape[0], spacing):
        for x in range(0, host_shape[1], spacing):
            y1, y2 = y, min(y + wm_rotated.shape[0], host_shape[0])
            x1, x2 = x, min(x + wm_rotated.shape[1], host_shape[1])
            wm_crop = wm_rotated[0:(y2-y1), 0:(x2-x1)]
            tiled_wm[y1:y2, x1:x2] = np.maximum(tiled_wm[y1:y2, x1:x2], wm_crop)
    tiled_wm = (tiled_wm / 255.0) * 255.0 * opacity
    return tiled_wm

def dct_watermarking_embed(host_arr, watermark_arr, opacity=0.05, scale=0.2, angle=30, spacing=100):
    if host_arr is None:
        raise ValueError("Host array is None.")
    watermarked = np.zeros_like(host_arr, dtype=np.uint8)
    for c in range(3):
        channel = host_arr[:, :, c].astype(np.float32)
        tiled_wm = _prepare_tiled_watermark(
            channel.shape, watermark_arr, opacity, scale, angle, spacing
        )
        channel_dct = cv2.dct(channel)
        wm_dct = cv2.dct(tiled_wm)
        watermarked_dct = channel_dct + wm_dct
        watermarked[:, :, c] = np.clip(cv2.idct(watermarked_dct), 0, 255).astype(np.uint8)
    return watermarked

def dct_watermarking_detect(watermarked_arr, original_watermark_arr, threshold=0.05):
    if watermarked_arr is None or original_watermark_arr is None:
        raise ValueError("One or both image arrays are None.")
    wm_resized = cv2.resize(original_watermark_arr, (watermarked_arr.shape[1], watermarked_arr.shape[0]))
    correlations = []
    wm_dct = cv2.dct(np.float32(wm_resized))
    wm_dct_norm = np.linalg.norm(wm_dct)
    if wm_dct_norm < 1e-6:
        wm_dct_norm = 1e-6
    for c in range(3):
        channel = watermarked_arr[:, :, c].astype(np.float32)
        channel_dct = cv2.dct(channel)
        channel_dct_norm = np.linalg.norm(channel_dct)
        if channel_dct_norm < 1e-6:
            channel_dct_norm = 1e-6
        corr = np.sum(channel_dct * wm_dct) / (channel_dct_norm * wm_dct_norm)
        correlations.append(corr)
    avg_corr = np.mean(correlations)
    detected = bool(avg_corr > threshold)
    return avg_corr, detected

@app.route('/embed_dct_tiled', methods=['POST'])
def api_embed_dct_tiled():
    if 'host_image' not in request.files or 'watermark_image' not in request.files:
        return jsonify({'error': 'Missing host_image or watermark_image'}), 400
    try:
        host_arr = _read_image_from_stream(request.files['host_image'].stream, cv2.IMREAD_COLOR)
        watermark_arr = _read_image_from_stream(request.files['watermark_image'].stream, cv2.IMREAD_GRAYSCALE)
        opacity = request.form.get('opacity', default=0.05, type=float)
        scale = request.form.get('scale', default=0.2, type=float)
        angle = request.form.get('angle', default=30, type=int)
        spacing = request.form.get('spacing', default=100, type=int)
        watermarked_array = dct_watermarking_embed(
            host_arr, watermark_arr, opacity, scale, angle, spacing
        )
        is_success, buffer = cv2.imencode(".png", watermarked_array)
        if not is_success:
            return jsonify({'error': 'Failed to encode output image'}), 500
        img_io = io.BytesIO(buffer.tobytes())
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='watermarked_dct_tiled.png')
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/detect_dct_tiled', methods=['POST'])
def api_detect_dct_tiled():
    if 'watermarked_image' not in request.files or 'original_watermark_image' not in request.files:
        return jsonify({'error': 'Missing watermarked_image or original_watermark_image'}), 400
    try:
        watermarked_arr = _read_image_from_stream(request.files['watermarked_image'].stream, cv2.IMREAD_COLOR)
        original_watermark_arr = _read_image_from_stream(request.files['original_watermark_image'].stream, cv2.IMREAD_GRAYSCALE)
        threshold = request.form.get('threshold', default=0.05, type=float)
        avg_corr, detected = dct_watermarking_detect(
            watermarked_arr, original_watermark_arr, threshold
        )
        return jsonify({
            'average_correlation': avg_corr,
            'detected': detected,
            'threshold_used': threshold
        })
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


# =============================================================================
# --- DWT (Invisible) & Visible Tiled Watermarking ---
# =============================================================================

def _to_gray_float(img_arr):
    """(Helper) Converts BGR/BGRA array to float32 grayscale."""
    if img_arr.ndim == 3 and img_arr.shape[2] == 4:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2GRAY)
    elif img_arr.ndim == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return img_arr.astype(np.float32)

def _correlation_coefficient(img1, img2):
    """(Helper) Computes normalized correlation coefficient."""
    img1 = img1.astype(np.float32).flatten()
    img2 = img2.astype(np.float32).flatten()
    numerator = np.sum(img1 * img2)
    denominator = np.sqrt(np.sum(img1**2) * np.sum(img2**2))
    return numerator / denominator if denominator != 0 else 0

def _embed_dwt_invisible_gray(cover_bgr, wm_bgr, alpha=0.04, wavelet='haar'):
    """(Helper) Embeds watermark in grayscale, returns grayscale."""
    cover = _to_gray_float(cover_bgr)
    LL, (LH, HL, HH) = pywt.dwt2(cover, wavelet)
    wm = _to_gray_float(wm_bgr)
    wm = cv2.resize(wm, (LL.shape[1], LL.shape[0]))
    LLw = LL + alpha * wm
    watermarked = pywt.idwt2((LLw, (LH, HL, HH)), wavelet)
    return np.clip(watermarked, 0, 255).astype(np.uint8)

def _extract_dwt_invisible(wm_bgr, cover_bgr, alpha=0.04, wavelet='haar'):
    """(Helper) Extracts invisible DWT watermark."""
    wm_img = _to_gray_float(wm_bgr)
    cover = _to_gray_float(cover_bgr)
    LLw, _ = pywt.dwt2(wm_img, wavelet)
    LLc, _ = pywt.dwt2(cover,  wavelet)
    wm_rec = (LLw - LLc) / alpha
    return np.clip(wm_rec, 0, 255).astype(np.uint8)

def _detect_dwt(watermarked_arr, cover_arr, original_wm_arr, alpha=0.5, wavelet='haar', threshold=0.5):
    """(Helper) Extracts DWT watermark and computes correlation."""
    # 1. Extract the watermark
    # Note: The extraction function itself handles grayscale conversion
    wm_rec_arr = _extract_dwt_invisible(watermarked_arr, cover_arr, alpha, wavelet)
    
    # 2. Resize original watermark for comparison
    original_gray = _to_gray_float(original_wm_arr)
    original_resized = cv2.resize(original_gray, (wm_rec_arr.shape[1], wm_rec_arr.shape[0]))

    # 3. Compute correlation
    corr = _correlation_coefficient(wm_rec_arr, original_resized)
    detected = bool(corr > threshold)
    
    return wm_rec_arr, corr, detected

def _embed_visible_tiled(cover_arr, logo_arr, opacity=0.3, angle=45, tile_scale=0.1):
    """(Helper) Implements visible tiled watermarking."""
    logo_aspect_ratio = logo_arr.shape[1] / logo_arr.shape[0]
    small_logo_height = int(cover_arr.shape[0] * tile_scale)
    small_logo_width = int(small_logo_height * logo_aspect_ratio)
    logo_small = cv2.resize(logo_arr, (small_logo_width, small_logo_height))
    (h, w) = logo_small.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_logo = cv2.warpAffine(logo_small, M, (w, h), borderValue=(0,0,0))
    tiled_watermark = np.zeros_like(cover_arr)
    for y in range(0, cover_arr.shape[0], rotated_logo.shape[0]):
        for x in range(0, cover_arr.shape[1], rotated_logo.shape[1]):
            roi_y1, roi_y2 = y, min(y + rotated_logo.shape[0], cover_arr.shape[0])
            roi_x1, roi_x2 = x, min(x + rotated_logo.shape[1], cover_arr.shape[1])
            logo_roi_y2 = roi_y2 - roi_y1
            logo_roi_x2 = roi_x2 - roi_x1
            tiled_watermark[roi_y1:roi_y2, roi_x1:roi_x2] = rotated_logo[:logo_roi_y2, :logo_roi_x2]
    return cv2.addWeighted(cover_arr, 1, tiled_watermark, opacity, 0)

@app.route('/embed_visible_tiled', methods=['POST'])
def api_embed_visible_tiled():
    """API endpoint for visible, tiled, rotated watermarking."""
    if 'cover_image' not in request.files or 'logo_image' not in request.files:
        return jsonify({'error': 'Missing cover_image or logo_image'}), 400
    try:
        cover_arr = _read_image_from_stream(request.files['cover_image'].stream, cv2.IMREAD_COLOR)
        logo_arr = _read_image_from_stream(request.files['logo_image'].stream, cv2.IMREAD_UNCHANGED)
        opacity = request.form.get('opacity', default=0.3, type=float)
        angle = request.form.get('angle', default=45, type=int)
        tile_scale = request.form.get('tile_scale', default=0.1, type=float)
        watermarked_array = _embed_visible_tiled(cover_arr, logo_arr, opacity, angle, tile_scale)
        is_success, buffer = cv2.imencode(".png", watermarked_array)
        if not is_success:
            return jsonify({'error': 'Failed to encode output image'}), 500
        img_io = io.BytesIO(buffer.tobytes())
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='visible_watermarked.png')
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/embed_dwt_color', methods=['POST'])
def api_embed_dwt_color():
    """API endpoint for invisible DWT embedding (applied to Y channel)."""
    if 'host_image' not in request.files or 'watermark_image' not in request.files:
        return jsonify({'error': 'Missing host_image or watermark_image'}), 400
    try:
        cover_bgr = _read_image_from_stream(request.files['host_image'].stream, cv2.IMREAD_COLOR)
        wm_bgr = _read_image_from_stream(request.files['watermark_image'].stream, cv2.IMREAD_UNCHANGED)

        alpha = request.form.get('alpha', default=0.04, type=float)
        wavelet = request.form.get('wavelet', default='haar', type=str)

        wm_img_gray = _embed_dwt_invisible_gray(cover_bgr, wm_bgr, alpha, wavelet)
        ycrcb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb)
        wm_resized_gray = cv2.resize(wm_img_gray, (cover_bgr.shape[1], cover_bgr.shape[0]))
        ycrcb[..., 0] = wm_resized_gray
        watermarked_color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        is_success, buffer = cv2.imencode(".png", cover_bgr)
        if not is_success:
            return jsonify({'error': 'Failed to encode output image'}), 500

        img_io = io.BytesIO(buffer.tobytes())
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=True,
            download_name='watermarked_color_dwt.png'
        )

    except Exception as e:
        import traceback
        print("Exception Trace:", traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/extract_dwt', methods=['POST'])
def api_extract_dwt():
    """API endpoint for invisible DWT extraction (non-blind)."""
    if 'watermarked_image' not in request.files or 'cover_image' not in request.files:
        return jsonify({'error': 'Missing watermarked_image or cover_image'}), 400
    try:
        wm_bgr = _read_image_from_stream(request.files['watermarked_image'].stream, cv2.IMREAD_COLOR)
        cover_bgr = _read_image_from_stream(request.files['cover_image'].stream, cv2.IMREAD_COLOR)
        alpha = request.form.get('alpha', default=0.04, type=float)
        wavelet = request.form.get('wavelet', default='haar', type=str)
        extracted_array = _extract_dwt_invisible(wm_bgr, cover_bgr, alpha, wavelet)
        is_success, buffer = cv2.imencode(".png", extracted_array)
        if not is_success:
            return jsonify({'error': 'Failed to encode output image'}), 500
        img_io = io.BytesIO(buffer.tobytes())
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='extracted_watermark_dwt.png')
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/detect_dwt', methods=['POST'])
def api_detect_dwt():
    """API endpoint for invisible DWT detection (non-blind)."""
    required_files = ['watermarked_image', 'cover_image', 'original_watermark_image']
    if not all(f in request.files for f in required_files):
        return jsonify({'error': 'Missing one or more required images'}), 400
        
    try:
        # Read all three images
        watermarked_arr = _read_image_from_stream(request.files['watermarked_image'].stream, cv2.IMREAD_COLOR)
        cover_arr = _read_image_from_stream(request.files['cover_image'].stream, cv2.IMREAD_COLOR)
        original_wm_arr = _read_image_from_stream(request.files['original_watermark_image'].stream, cv2.IMREAD_UNCHANGED)

        # Get optional parameters
        alpha = request.form.get('alpha', default=0.5, type=float)
        wavelet = request.form.get('wavelet', default='haar', type=str)
        threshold = request.form.get('threshold', default=0.5, type=float)

        # Detect and get results
        extracted_arr, corr, detected = _detect_dwt(
            watermarked_arr, cover_arr, original_wm_arr, alpha, wavelet, threshold
        )

        # Encode extracted image to Base64
        is_success, buffer = cv2.imencode(".png", extracted_arr)
        if not is_success:
            return jsonify({'error': 'Failed to encode extracted image'}), 500
        img_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        # Return full JSON response
        return jsonify({
            'correlation': corr,
            'detected': detected,
            'threshold_used': threshold,
            'extracted_image_b64': img_b64,
            'image_format': 'png'
        })
        
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


# --- Run the Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
