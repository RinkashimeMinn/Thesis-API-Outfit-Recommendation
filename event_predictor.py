import os
import io
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image as PILImage, Image as PILRaw
from siamese_network import SiameseNetwork
from google.cloud import storage
from google.oauth2 import service_account
import gdown

SERVICE_ACCOUNT_PATH = "/etc/secrets/morphfit-key"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_PATH
)
storage_client = storage.Client(
    credentials=credentials,
    project=credentials.project_id
)
gcs_bucket = storage_client.bucket("morphfit-thesis")
# ✅ Google Drive Model Setup
MODEL_PATH = "siamese_model.pt"
GOOGLE_DRIVE_ID = "1b33sVOOrKvb7fFQieD-oMW7Q41hckbBD"  # Your model file ID

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}", MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

# ✅ Load the model with download logic
download_model_if_needed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

EVENT_LABELS = [
    "Job Interviews", "Birthday", "Graduations", "MET Gala", "Business Meeting",
    "Beach", "Picnic", "Summer", "Funeral", "Romantic Dinner",
    "Cold", "Casual", "Wedding"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_event_from_filenames(category_filename_pairs):
    """
    Returns just the top-3 and full probabilities for each event.
    """
    category_order = ["Hats","Accessories","Sunglasses","Outerwear","All-body/Tops","Bottoms","Shoes"]
    CATEGORY_RENAME = {"All-body": "All-body/Tops", "Tops": "All-body/Tops"}
    category_map = {CATEGORY_RENAME.get(cat, cat): fname for cat, fname in category_filename_pairs}

    # Build input batch
    input_batch = []
    for slot in category_order:
        if slot in category_map:
            blob = gcs_bucket.blob(category_map[slot])
            data = blob.download_as_bytes()
            img = PILImage.open(io.BytesIO(data)).convert("RGB")
            input_batch.append(transform(img).unsqueeze(0).to(device))
        else:
            blank = PILRaw.new("RGB", (224, 224), (255, 255, 255))
            input_batch.append(transform(blank).unsqueeze(0).to(device))

        # Forward pass
    with torch.no_grad():
        outputs = model(*input_batch)
        logits = outputs[0]
        # If output is batched (e.g., shape [1, num_events]), take first element
        if logits.ndim > 1:
            logits = logits[0]

    # Compute probabilities over the event dimension
    probs = F.softmax(logits, dim=0).cpu().numpy()

    # Convert to Python list of floats
    prob_list = probs.tolist()

    probs = F.softmax(logits, dim=0).cpu().numpy()
    # Ensure probs is a flat list of floats
    prob_list = [float(p) for p in probs.tolist()]
    sorted_probs = sorted(zip(EVENT_LABELS, prob_list), key=lambda x: x[1], reverse=True)

    # Prepare output
    return {
        'top_3_predictions': [
            {'event': event, 'probability': score}
            for event, score in sorted_probs[:3]
        ],
        'all_event_probabilities': [
            {'event': event, 'probability': score}
            for event, score in sorted_probs
        ]
    }
