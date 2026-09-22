# Camera-Based Sudoku Solver

A real-time Sudoku solver that reads a puzzle through a webcam and overlays the solution directly onto the live video feed.

## How It Works

1. **Grid Detection** — OpenCV locates the Sudoku grid in the camera frame via contour detection, then applies a perspective warp to get a flat, top-down view of the board.
2. **Digit Recognition** — Each of the 81 cells is cropped and passed through a custom-trained CNN (PyTorch) to classify the digit, or detect that a cell is empty.
3. **Solving** — A backtracking solver validates and solves the recognized board.
4. **Live Overlay** — The solved digits are rendered onto a blank canvas, then warped back into the camera's perspective using the inverse of the original transform, so the solution appears directly on top of the real grid — in green, only in the cells that were originally blank.

## Tech Stack

- **Python**
- **OpenCV** — grid detection, perspective transforms, live video pipeline
- **PyTorch** — CNN for digit classification
- **PIL** — synthetic training data generation

## Key Technical Challenges

**MNIST didn't generalize to printed puzzles.**
The digit-recognition model was initially trained on MNIST, but MNIST is handwritten digits — its stroke shape, thickness, and style don't match the clean printed fonts on an actual Sudoku puzzle. This caused systematic misreads on real camera input. The fix: generate a synthetic dataset of printed digits (multiple system fonts, randomized rotation, affine transforms, and motion-blur simulation via PIL), then train on a combined MNIST + synthetic-printed dataset. This brought recognition accuracy to ~98.5% on real photographed digits.

**Keeping the overlay locked to the grid during camera movement.**
The solved-digit overlay is computed by inverse-warping a solution canvas using that frame's perspective transform. Early versions recomputed this only once, causing the overlay to drift out of alignment as soon as the camera moved. The fix was to recompute the inverse warp every frame using the current frame's transform — and to handle frames where grid detection briefly fails (e.g. during motion blur) by falling back to the last successfully detected transform for a short grace period, instead of dropping the overlay outright. This keeps the solution visually locked onto the physical grid, with brief, graceful tolerance for detection gaps rather than hard flicker.

## Pipeline Files

| File | Purpose |
|---|---|
| `grid_detector.py` | Grid contour detection, corner ordering, perspective warp |
| `train_printed_digit_model.py` | Trains the CNN on combined MNIST + synthetic printed digits |
| `generate_printed_digits.py` | Generates synthetic printed-digit training data |
| `sudoku_pipeline.py` | Splits the warped grid into cells and recognizes each digit |
| `solver.py` | Backtracking Sudoku solver and board validation |
| `live_pipeline.py` | Main live webcam loop — detection, recognition, solving, overlay |

## Running It

```bash
python live_pipeline.py
```

Hold a Sudoku puzzle up to your webcam. Once the grid is detected and held steady, the board is recognized and solved automatically, with the solution overlaid live in green.
