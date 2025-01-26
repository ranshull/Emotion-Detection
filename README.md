# Emotion-Detection

This is a simple emotion recognition web application built with **Flask** and **TensorFlow Keras**. The app allows users to upload an image, predicts the emotion expressed in the image, and provides feedback based on the recognized emotion.

## Features
- Predicts emotions from facial images using a pre-trained emotion recognition model.
- Displays feedback based on the detected emotion.
- Supports 7 emotion labels: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

## Requirements
- Python 3.x
- Flask
- TensorFlow
- OpenCV
- Numpy

## Installation

1. Clone the repository or download the files.

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure that you have the pre-trained emotion model saved in the `models` directory. If not, you can place the trained model file (`emotion_model.h5`) in the `models` folder.

## Running the Application

1. Navigate to the project directory:
   ```bash
   cd path/to/project
   ```

2. Start the Flask app:
   ```bash
   python app.py
   ```

3. Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to use the application.

## How it Works

1. The user uploads a photo through the web interface.
2. The image is processed and resized to 48x48 pixels.
3. The model predicts the emotion based on the processed image.
4. The emotion label is returned along with a personalized feedback message.

### Supported Emotions:
- **Angry**: Take a few deep breaths and try to relax.
- **Disgust**: Remember to focus on the positives around you.
- **Fear**: Stay calm; face your fears one step at a time.
- **Happy**: Keep smiling! Happiness is contagious.
- **Neutral**: All set; stay balanced and steady.
- **Sad**: Consider talking to a friend or doing something you enjoy.
- **Surprise**: Embrace the unexpected; life is full of surprises!

## File Structure
```
/project-root
    /static           # Static files (images, CSS, etc.)
    /templates        # HTML templates
    /models           # Folder containing the emotion model
    app.py            # Flask application script
    requirements.txt  # List of dependencies
```

## License
This project is open-source and available under the MIT License.
```

You can now copy and paste this into your `README.md` file. Let me know if you need any adjustments!