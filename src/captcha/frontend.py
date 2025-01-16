import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
import io
import bentoml


def normalize(image: np.array) -> np.array:
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.
    Args:
        images (np.array): Image.
    Returns:
        np.array: Normalized image.
    """
    return (image - image.mean()) / image.std()


def get_backend_url():
    """Get the URL of the backend service."""
    # parent = "projects/dtumlops-447710/locations/europe-west1"
    # client = run_v2.ServicesClient()
    # services = client.list_services(parent=parent)
    # for service in services:
    #    if service.name.split("/")[-1] == "backend":
    #        return service.uri
    # return os.environ.get("BACKEND", None)
    return "https://backend-1048604560911.europe-west1.run.app"


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    # predict_url = f"{backend}/predict"
    with bentoml.SyncHTTPClient(backend) as client:
        response = client.predict(image=image)
        # print(respp)
    # response = requests.post(predict_url, files={"image": image}, timeout=10)
    # if response.status_code == 200:
    # return response.json()
    # return None
    return response


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Captcha Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        image = Image.open(io.BytesIO(image))
        # image = image.convert("L")  # Convert to grayscale
        # image = image.resize((28, 28))  # Resize to match the minimum input size of the model
        image = np.array(image)
        image = np.expand_dims(image, 2)
        image = np.transpose(image, (2, 0, 1))  # Change to CHW format
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = normalize(image)
        result = classify_image(image, backend=backend)
        image = image.squeeze(0).squeeze(0)

        if result is not None:
            result = result.squeeze(0)
            result = np.exp(result - max(result)) / np.sum(np.exp(result - max(result)))
            class_names = np.array(
                ["2", "3", "4", "5", "6", "7", "8", "a", "b", "c", "d", "e", "f", "g", "m", "n", "p", "w", "x", "y"]
            )
            # prediction = result["prediction"]
            # probabilities = result["probabilities"]

            # show the image and prediction
            image = (image - image.min()) / (image.max() - image.min())
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", class_names[np.argmax(result)])

            # make a nice bar chart
            data = {"Class": [f"Class {c}" for c in class_names], "Probability": result}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
