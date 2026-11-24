"""Flask web server for the Emotion Detector application."""

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector

app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def emotion_detection():
    """Analyze submitted text and return the emotion scores."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return "Invalid text! Please try again!", 400
    response = emotion_detector(text_to_analyze)
    if response["dominant_emotion"] is None:
        return "Invalid text! Please try again!", 400
    values = ", ".join(
        f"'{emotion}': {response[emotion]}"
        for emotion in ["anger", "disgust", "fear", "joy", "sadness"]
    )
    return (
        f"For the given statement, the system response is {values}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )


@app.route("/")
def render_index_page():
    """Render the application page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
