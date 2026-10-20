
import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image

def predict_video(video_input):
    if video_input is None:
        return "Please upload a video first."

    # Gradio may provide a file path or a dictionary
    if isinstance(video_input, dict):
        video_path = (
            video_input.get("path")
            or video_input.get("name")
        )
    else:
        video_path = video_input

    if not video_path:
        return "Could not find the uploaded video."

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Error: Unable to open this video."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return "Error: The video has no readable frames."

    # Sample up to 12 frames across the video
    sample_count = min(12, total_frames)
    positions = np.linspace(
        0, total_frames - 1, sample_count, dtype=int
    )

    frames = []

    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        success, frame = cap.read()

        if success:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

    cap.release()

    if not frames:
        return "Error: Could not extract video frames."

    try:
        # Analyze every sampled frame
        results = detector(frames)

        fake_scores = []

        for result in results:
            scores = {
                item["label"].lower(): float(item["score"])
                for item in result
            }

            # Match labels returned by the model
            if "fake" in scores:
                fake_score = scores["fake"]
            elif "deepfake" in scores:
                fake_score = scores["deepfake"]
            elif "real" in scores:
                fake_score = 1.0 - scores["real"]
            else:
                return (
                    "Unrecognized model labels: "
                    + ", ".join(scores.keys())
                )

            fake_scores.append(fake_score)

        average_fake = float(np.mean(fake_scores))

        if average_fake >= 0.5:
            verdict = "LIKELY FAKE"
            confidence = average_fake
        else:
            verdict = "LIKELY REAL"
            confidence = 1.0 - average_fake

        return (
            f"Prediction: {verdict}\n\n"
            f"Model score for this prediction: "
            f"{confidence:.2%}\n"
            f"Frames analyzed: {len(frames)}\n\n"
            "Warning: This is an AI estimate, not proof. "
            "Accuracy may vary for videos outside the "
            "model's training data."
        )

    except Exception as error:
        return (
            "Prediction failed.\n"
            f"Error: {str(error)}\n\n"
            "Check the model loading instructions and "
            "installed library versions."
        )


demo = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Textbox(
        label="Real or Fake Prediction",
        lines=7
    ),
    title="Multilingual Deepfake Detection",
    description=(
        "Upload a video to analyze sampled frames "
        "using a pretrained deepfake detection model."
    )
)

if __name__ == "__main__":
    demo.launch()
