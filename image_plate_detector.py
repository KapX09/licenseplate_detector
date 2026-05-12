import cv2  
import numpy as np
import onnxruntime as ort
import pytesseract
import sys
from pathlib import Path


MODEL_PATH  = "models/best.onnx"
TEST_DIR    = "test_images"
CONF_THRESH = 0.4
IOU_THRESH  = 0.45
INPUT_SIZE  = (640, 640)
SKIP_LEFT   = 0.10

TESS_CONFIG = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

LETTER_LIKE_DIGITS = {'0':'O', '1':'I', '5':'S', '8':'B'}
DIGIT_LIKE_LETTERS = {'O':'0', 'I':'1', 'S':'5', 'B':'8', 'Z':'2', 'G':'6'}

#ONNX session
sess_opts = ort.SessionOptions()
sess_opts.log_severity_level = 3
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.intra_op_num_threads = 4
sess_opts.inter_op_num_threads = 2

cuda_opts = {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}
session = ort.InferenceSession(
    MODEL_PATH, sess_opts=sess_opts,
    providers=[('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']
)
input_name = session.get_inputs()[0].name


#detection helpers
def preprocess(img_bgr, size=INPUT_SIZE):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, size)
    blob = resized.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
    return blob, w, h


def xywh2xyxy(boxes):
    out = boxes.copy()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def nms(boxes, scores, iou_thresh):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2]-boxes[i, 0]) * (boxes[i, 3]-boxes[i, 1])
        area_j = (boxes[idxs[1:], 2]-boxes[idxs[1:], 0]) * \
                 (boxes[idxs[1:], 3]-boxes[idxs[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        idxs = idxs[1:][iou < iou_thresh]
    return keep


def postprocess(output, orig_w, orig_h, size=INPUT_SIZE):
    pred = output[0]
    if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
        pred = pred[0].T
    else:
        pred = pred[0]

    boxes_xywh = pred[:, :4]
    scores     = pred[:, 4]
    mask       = scores >= CONF_THRESH
    boxes_xywh = boxes_xywh[mask]
    scores     = scores[mask]

    if len(scores) == 0:
        return []

    boxes_xyxy = xywh2xyxy(boxes_xywh)
    keep       = nms(boxes_xyxy, scores, IOU_THRESH)
    boxes_xyxy = boxes_xyxy[keep]
    scores     = scores[keep]

    sx, sy = orig_w / size[0], orig_h / size[1]
    results = []
    for box, conf in zip(boxes_xyxy, scores):
        x1 = max(0, int(box[0] * sx))
        y1 = max(0, int(box[1] * sy))
        x2 = min(orig_w, int(box[2] * sx))
        y2 = min(orig_h, int(box[3] * sy))
        results.append([x1, y1, x2, y2, float(conf)])
    return results


#OCR helpers
def enhance(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.std() < 40:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
    # otsu binarize adapts to each crop automatically
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def upscale(gray, target_h=64):
    h, w = gray.shape[:2]
    if h < target_h:
        scale = target_h / h
        gray = cv2.resize(gray, (int(w * scale), target_h),
                          interpolation=cv2.INTER_CUBIC)
    return gray


def deskew(gray):
    # correct slight plate tilt before OCR
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 1:
        return gray  # skiping if tilt negligible
    M = cv2.getRotationMatrix2D(
        (gray.shape[1] // 2, gray.shape[0] // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                          borderValue=255)


def smart_clean(text):
    if not text:
        return ""
    text = "".join(c for c in text.upper() if c.isalnum())

    if len(text) < 4:
        return ""

    result = list(text)

    # first 2 must be letters
    for i in range(min(2, len(result))):
        if result[i].isdigit():
            result[i] = LETTER_LIKE_DIGITS.get(result[i], result[i])

    # positions 2-3 must be digits
    for i in range(2, min(4, len(result))):
        if result[i].isalpha():
            result[i] = DIGIT_LIKE_LETTERS.get(result[i], result[i])

    # last 4 must be digits
    for i in range(max(4, len(result) - 4), len(result)):
        if result[i].isalpha():
            result[i] = DIGIT_LIKE_LETTERS.get(result[i], result[i])

    return "".join(result)


def remove_phantom(text):
    import re
    # strip leading char if first two chars aren't both letters
    while len(text) > 2 and not re.match(r'^[A-Z]{2}', text):
        text = text[1:]
    return text


def tess_read(gray):
    data = pytesseract.image_to_data(
        gray, config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )
    words = [w for w, c in zip(data['text'], data['conf'])
             if w.strip()]
    confs = [int(c) for c in data['conf']
             if c != '-1']
    text  = "".join(words)
    conf  = sum(confs) / len(confs) / 100 if confs else 0.0
    return text, conf


# multi-line aware plate reader
def read_plate(img_bgr, x1, y1, x2, y2):
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "", 0.0

    h, w = crop.shape[:2]

    # trim left badge + right bleed
    left  = int(w * SKIP_LEFT)
    right = max(3, int(w * 0.04))
    crop  = crop[:, left:w - right]

    gray = enhance(crop)
    gray = upscale(gray, target_h=64)
    gray = deskew(gray)

    # detect if two-line plate by aspect ratio
    ch, cw = gray.shape[:2]
    is_two_line = (ch / cw) > 0.45  # taller than wide = two lines likely

    if is_two_line:
        # split into top and bottom half, read each separately
        mid  = ch // 2
        top  = gray[:mid, :]
        bot  = gray[mid:, :]
        t1, c1 = tess_read(top)
        t2, c2 = tess_read(bot)
        text = t1 + t2
        conf = (c1 + c2) / 2
    else:
        text, conf = tess_read(gray)

    text = remove_phantom(text)
    text = smart_clean(text)
    return text, conf


#main
def process_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] cannot read {img_path.name}")
        return

    blob, orig_w, orig_h = preprocess(img)
    output     = session.run(None, {input_name: blob})
    detections = postprocess(output, orig_w, orig_h)

    print(f"\n{img_path.name}  →  {len(detections)} plate(s) found")

    for i, (x1, y1, x2, y2, det_conf) in enumerate(detections):
        text, ocr_conf = read_plate(img, x1, y1, x2, y2)
        label = f"{text}  (det:{det_conf:.2f} ocr:{ocr_conf:.2f})"
        print(f"  [{i+1}] {label}")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow(img_path.name, img)
    print("  Press any key for next image, 'q' to quit ...")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(img_path.name)
    if key == ord("q"):
        sys.exit(0)


if __name__ == "__main__":
    test_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(TEST_DIR)
    exts     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images   = [p for p in sorted(test_dir.iterdir())
                if p.suffix.lower() in exts]

    if not images:
        print(f"No images found in '{test_dir}'")
        sys.exit(1)

    print(f"Processing {len(images)} image(s) from '{test_dir}' ...")
    for img_path in images:
        process_image(img_path)

    print("\nDone.")


'''
Processing 11 image(s) from 'test_images' ...

img1.jpg  →  1 plate(s) found
  [1]   (det:0.86 ocr:-0.01)
  Press any key for next image, 'q' to quit ...

img10.jpg  →  1 plate(s) found
  [1] KL07BF5000  (det:0.70 ocr:0.17)
  Press any key for next image, 'q' to quit ...

img11.jpg  →  1 plate(s) found
  [1] AUQ819  (det:0.74 ocr:0.05)
  Press any key for next image, 'q' to quit ...

img2.jpg  →  1 plate(s) found
  [1] KA51NJ8156  (det:0.76 ocr:0.05)
  Press any key for next image, 'q' to quit ...

img3.jpg  →  1 plate(s) found
  [1] UK07BA7252  (det:0.77 ocr:0.14)
  Press any key for next image, 'q' to quit ...

img4.jpg  →  1 plate(s) found
  [1] UP78EJ7683  (det:0.77 ocr:0.11)
  Press any key for next image, 'q' to quit ...

img5.jpg  →  1 plate(s) found
  [1] DL10C64693  (det:0.72 ocr:0.08)
  Press any key for next image, 'q' to quit ...

img6.jpg  →  1 plate(s) found
  [1] KA51NJ8156  (det:0.76 ocr:0.05)
  Press any key for next image, 'q' to quit ...

img7.jpg  →  1 plate(s) found
  [1] WP12EC5111  (det:0.78 ocr:0.05)
  Press any key for next image, 'q' to quit ...

img8.jpg  →  1 plate(s) found
  [1] TS08ER164  (det:0.69 ocr:-0.01)
  Press any key for next image, 'q' to quit ...

img9.jpg  →  1 plate(s) found
  [1] AS01BZ2002  (det:0.81 ocr:0.02)
  Press any key for next image, 'q' to quit ...

Done.
'''
