import cv2
import numpy as np
import onnxruntime as ort
import easyocr
import sys
from pathlib import Path


MODEL_PATH  = "models/best.onnx"
TEST_DIR    = "test_images"
CONF_THRESH = 0.4
IOU_THRESH  = 0.45
INPUT_SIZE  = (640, 640)
SKIP_LEFT   = 0.10


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

reader = easyocr.Reader(["en"], gpu=True)


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
    scores = pred[:, 4]
    mask = scores >= CONF_THRESH
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]

    if len(scores) == 0:
        return []

    boxes_xyxy = xywh2xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, scores, IOU_THRESH)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    sx = orig_w / size[0]
    sy = orig_h / size[1]
    results = []
    for box, conf in zip(boxes_xyxy, scores):
        x1 = max(0, int(box[0] * sx))
        y1 = max(0, int(box[1] * sy))
        x2 = min(orig_w, int(box[2] * sx))
        y2 = min(orig_h, int(box[3] * sy))
        results.append([x1, y1, x2, y2, float(conf)])
    return results


def enhance_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.std() < 40:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return crop


def read_plate(img_bgr, x1, y1, x2, y2):
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    h, w = crop.shape[:2]
    left_skip = int(w * SKIP_LEFT)
    inset = max(2, int(w * 0.02))
    crop = crop[:, left_skip:w - inset]

    if crop.shape[1] < 200:
        scale = 200 / crop.shape[1]
        crop = cv2.resize(crop, (200, int(crop.shape[0] * scale)),
                          interpolation=cv2.INTER_CUBIC)

    crop = enhance_crop(crop)

    results = reader.readtext(
        crop,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        paragraph=False,
        text_threshold=0.5,
        low_text=0.3,
        width_ths=1.0,
        height_ths=0.8,
        slope_ths=0.2,
        ycenter_ths=0.7,
    )

    if not results:
        return ""

    def y_center(r):
        return (r[0][0][1] + r[0][2][1]) / 2

    band_size = max(10, int(crop.shape[0] * 0.3))
    results.sort(key=lambda r: (round(y_center(r) / band_size), r[0][0][0]))
    return "".join(r[1] for r in results).upper()


def process_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] cannot read {img_path.name}")
        return

    blob, orig_w, orig_h = preprocess(img)
    output = session.run(None, {input_name: blob})
    detections = postprocess(output, orig_w, orig_h)

    print(f"\n{img_path.name}  →  {len(detections)} plate(s) found")

    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        text = read_plate(img, x1, y1, x2, y2)
        label = f"{text}  ({conf:.2f})"
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
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(test_dir.iterdir()) if p.suffix.lower() in exts]

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
  [1] HH2ABQ8  (0.86)
  Press any key for next image, 'q' to quit ...

img10.jpg  →  1 plate(s) found
  [1] KLOZBFSOOO  (0.70)
  Press any key for next image, 'q' to quit ...

img11.jpg  →  1 plate(s) found
  [1] 6J23FAU0819  (0.74)
  Press any key for next image, 'q' to quit ...

img2.jpg  →  1 plate(s) found
  [1] K451MJ8156  (0.76)
  Press any key for next image, 'q' to quit ...

img3.jpg  →  1 plate(s) found
  [1] UK078472527  (0.77)
  Press any key for next image, 'q' to quit ...

img4.jpg  →  1 plate(s) found
  [1] UP78EJ7683  (0.77)
  Press any key for next image, 'q' to quit ...

img5.jpg  →  1 plate(s) found
  [1] DL10C64693  (0.72)
  Press any key for next image, 'q' to quit ...

img6.jpg  →  1 plate(s) found
  [1] K451MJ8156  (0.76)
  Press any key for next image, 'q' to quit ...

img7.jpg  →  1 plate(s) found
  [1] UP3ZEC5141  (0.78)
  Press any key for next image, 'q' to quit ...

img8.jpg  →  1 plate(s) found
  [1] TSOBER1642  (0.69)
  Press any key for next image, 'q' to quit ...

img9.jpg  →  1 plate(s) found
  [1] AS01BZ20027  (0.81)
  Press any key for next image, 'q' to quit ...

Done.
'''
