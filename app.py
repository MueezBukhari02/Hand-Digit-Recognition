import cv2
import numpy as np
import tensorflow as tf

# ── Load model ──────────────────────────────────────────
model = tf.keras.models.load_model('model/digit_model.keras')
print("Model loaded!")

# ── Canvas settings ──────────────────────────────────────
CANVAS_SIZE = 500        # square canvas 400x400 pixels
BRUSH_SIZE  = 18         # thickness of drawing stroke

# Create blank black canvas
canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

# Mouse state
drawing = False          # is the mouse button held down?
last_x, last_y = -1, -1 # last mouse position

# Prediction result
prediction = None
confidence = None

# handling mouse events for drawing
def mouse_handler(event, x, y, flags, param):
    global drawing, last_x, last_y, canvas, prediction, confidence

    if event == cv2.EVENT_LBUTTONDOWN:
        # Mouse button pressed — start drawing
        drawing = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        # Mouse moving — draw if button is held
        if drawing:
            cv2.line(canvas, (last_x, last_y), (x, y),
                    (255, 255, 255), BRUSH_SIZE)
            last_x, last_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        # Mouse button released — stop drawing
        drawing = False
        last_x, last_y = -1, -1


# process the canvas and predict the digit
def predict_digit():
    global prediction, confidence

    # Step 1: Grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Step 2: Check if canvas is empty
    coords = cv2.findNonZero(gray)
    if coords is None:
        return

    # Step 3: Count connected components (separate blobs/strokes)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    num_components, _ = cv2.connectedComponents(binary)
    # num_components includes background (always 1), so actual strokes = num_components - 1
    actual_strokes = num_components - 1

    # Step 4: Crop to bounding box with padding
    x, y, w, h = cv2.boundingRect(coords)
    pad = 30
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(CANVAS_SIZE - x, w + 2 * pad)
    h = min(CANVAS_SIZE - y, h + 2 * pad)
    cropped = gray[y:y+h, x:x+w]

    # Step 5: Resize, normalize, reshape
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    input_img = normalized.reshape(1, 28, 28, 1)

    # Step 6: Predict
    predictions = model.predict(input_img, verbose=0)
    raw_confidence = np.max(predictions[0]) * 100
    predicted_digit = np.argmax(predictions[0])

    # Step 7: Calculate shape properties
    total_pixels = int(binary.sum()) // 255
    canvas_area = CANVAS_SIZE * CANVAS_SIZE
    fill_ratio = total_pixels / canvas_area
    aspect_ratio = w / h if h > 0 else 0

    # Entropy check — real digits have one dominant probability
    # Scribbles have probabilities spread across many digits
    probs = predictions[0]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    # Low entropy = confident = real digit
    # High entropy = confused = likely not a digit

    is_valid = (
        raw_confidence >= 85 and    # high confidence
        entropy < 0.5 and           # model is not confused
        actual_strokes <= 4 and     # not too many separate strokes
        0.15 <= aspect_ratio <= 4.0 # reasonable shape
    )

    if is_valid:
        prediction = predicted_digit
        confidence = raw_confidence
    else:
        prediction = -1
        confidence = raw_confidence

    print(f"Strokes: {actual_strokes} | Fill: {fill_ratio:.3f} | AR: {aspect_ratio:.2f} | Conf: {raw_confidence:.1f}% | Entropy: {entropy:.3f}")



# ── Main loop ────────────────────────────────────────────
cv2.namedWindow("Draw a Digit")
cv2.setMouseCallback("Draw a Digit", mouse_handler)

print("Controls: SPACE = predict | C = clear | Q = quit")

while True:
    # Make a display copy so we can write text on it
    # without permanently modifying the canvas
    display = canvas.copy()

    # Draw instructions on screen
    cv2.putText(display, "SPACE: Predict  |  C: Clear  |  Q: Quit",
            (10, CANVAS_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Show prediction result if available
    if prediction is not None:
        if prediction == -1:
            # Not a digit — show in red
            cv2.putText(display, "Not a digit!",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.putText(display, f"Confidence: {confidence:.1f}%",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Valid digit — show in green
            cv2.putText(display, f"Digit: {prediction}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(display, f"Confidence: {confidence:.1f}%",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
    cv2.imshow("Draw a Digit", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    # Check if window was closed with X button
    elif cv2.getWindowProperty("Draw a Digit", cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord('c'):
        # Clear canvas and reset prediction
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
        prediction = None
        confidence = None
    elif key == 32:  # SPACE
        predict_digit()

cv2.destroyAllWindows()