from __future__ import annotations

import bentoml
import numpy as np
from onnxruntime import InferenceSession
from google.cloud import storage
import datetime
import json


@bentoml.service(workers=2)
class CaptchaClassifierService:
    """Captcha classifier service using ONNX model."""

    def __init__(self) -> None:
        self.model = InferenceSession("onnx_model.onnx")

    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        max_batch_size=128,
        max_latency_ms=1000,
    )
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict the class of the input image."""
        output = self.model.run(None, {"input": image.astype(np.float32)})
        class_names = np.array(
            ["2", "3", "4", "5", "6", "7", "8", "a", "b", "c", "d", "e", "f", "g", "m", "n", "p", "w", "x", "y"]
        )
        self.save_prediction_to_gcp(image, output[0][0].tolist(), class_names[np.argmax(output[0][0])])
        return output[0]

    def save_prediction_to_gcp(self, image: np.ndarray, outputs: list[float], prediction: str) -> None:
        """Save the prediction results to GCP bucket."""
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket("mlops_captcha_monitoring")
        time = datetime.datetime.now(tz=datetime.UTC)
        # Prepare prediction data
        image = image.squeeze(0).squeeze(0)
        image = image.flatten().tolist()
        data = {
            "image": image,
            "prediction": prediction,
            "probability": outputs,
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        }
        blob = bucket.blob(f"prediction_{time}.json")
        blob.upload_from_string(json.dumps(data))
        print("Prediction saved to GCP bucket.")
