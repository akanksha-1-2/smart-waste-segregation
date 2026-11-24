# Smart Waste Segregation

A recycling assistant that uses a camera and ONNX-based object detectors to identify waste items and tell the user which bin to use. It runs on a laptop and on a Raspberry Pi 4 Model B.

Built for the Bharat AI-SoC Challenge (2nd Edition), Problem Statement 2.

## Hardware

- Raspberry Pi 4 Model B (used for the Pi version, with a 16x2 I2C LCD)
- 16x2 I2C LCD with PCF8574 backpack (address `0x27`)
- Laptop (Windows) for the laptop version, with a display window
- Camera: Android phone running DroidCam over Wi-Fi
- Speaker (optional, for voice output)

I did not have a USB webcam, so I used a phone camera through DroidCam. The code also works with a USB webcam; only the camera source needs to change.

## Software

- Raspberry Pi OS Lite (64-bit) on the Pi
- Python 3.11.2 in a virtual environment on the Pi
- numpy, onnxruntime, opencv-python-headless, RPLCD, smbus2 (Pi)
- numpy, onnxruntime, opencv-python (laptop)
- espeak-ng (optional, for voice)
- YOLOv5 and YOLOv8 for training, run on Google Colab

## Files

```
smart-waste-segregation/
├── onnx_pi.py          main program for Raspberry Pi
├── onnx_live.py        laptop version with a display window
├── weights/
│   ├── best_ep33.onnx  cardboard, paper
│   ├── best_50.onnx    ewaste, metal
│   └── best_v8.onnx    plastic
├── .gitignore
└── README.md
```

## Models

I trained two models myself on Google Colab:

- `best_ep33.onnx` detects cardboard and paper (YOLOv5).
- `best_50.onnx` detects ewaste and metal (YOLOv5).

## How a Decision Is Made

1. Each frame is passed through the models.
2. A detection is kept only if it comes from the model assigned to its class.
3. Non-maximum suppression removes overlapping boxes.
4. Each class must meet a minimum confidence threshold.
5. Over a 1.5 second window, a class must appear in at least 3 frames and in at least half of all frames to count as the result.

| Class | Bin |
|---|---|
| plastic | PLASTIC BIN |
| paper | PAPER BIN |
| cardboard | PAPER BIN |
| metal | METAL BIN |
| ewaste | E-WASTE BIN |

## Setup

On the Raspberry Pi:

```bash
python3 -m venv ~/waste-env
source ~/waste-env/bin/activate
pip install numpy onnxruntime opencv-python-headless RPLCD smbus2
git clone https://github.com/akanksha-1-2/smart-waste-segregation.git
cd smart-waste-segregation
```

Enable I2C with `sudo raspi-config` (Interface Options → I2C), then reboot.

On the laptop:

```bash
py -m venv waste-env
waste-env\Scripts\activate
pip install numpy onnxruntime opencv-python
```

For DroidCam, install the app on the phone, connect the phone and the computer to the same Wi-Fi, and set the IP in `PHONE_URL` (or `CAMERA_SOURCE`) inside the code.

## Running

On the Raspberry Pi:

```bash
source ~/waste-env/bin/activate
cd ~/smart-waste-segregation
python onnx_pi.py
```

On the laptop (shows a window instead of the LCD):

```bash
python onnx_live.py
```

Decisions are printed in the terminal and logged to `output/log.csv`, with photos saved in `output/`.

## Offline Operation

The system does not need internet to run. With DroidCam, the phone and the computer only need to be on the same local Wi-Fi. With a USB webcam, no Wi-Fi is needed.

## Demo Videos

- Laptop with phone camera (DroidCam), live detection shown on the laptop: https://drive.google.com/file/d/1IE_EfrtlU76fKu7vt17e-U4uK8Ma0mYt/view?usp=sharing
- Raspberry Pi with LCD result: https://drive.google.com/file/d/1iEqWksqjgyystPvrumRmWFMWDypp8izU/view?usp=sharing

## Limitations

- Testing so far has used DroidCam over Wi-Fi, so frame rate depends on the network.
- The USB webcam setup has not been tested yet.
- Inference runs on the CPU only.
- The system can only recognise the classes it was trained on.

## Future Work

- Move inference to the PYNQ-ZU board.
- Add a servo motor that opens the correct bin lid after a decision.

## Author

Akanksha Pandey

## License

All rights reserved.
