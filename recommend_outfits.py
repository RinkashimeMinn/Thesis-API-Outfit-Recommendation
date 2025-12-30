import os
import json
import torch
from itertools import product, combinations
from PIL import Image as PILImage
from io import BytesIO
from torchvision import transforms
from flask_sqlalchemy import SQLAlchemy
from siamese_network import SiameseNetwork
from database import db, ImageModel, RecommendationResult, GeneratedOutfit
from tqdm import tqdm
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

# ✅ Step 1: Download model if missing
MODEL_PATH = "siamese_model.pt"
GOOGLE_DRIVE_ID = "1b33sVOOrKvb7fFQieD-oMW7Q41hckbBD"  # Replace with your actual file ID

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}", MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

download_model_if_needed()

# ✅ Step 2: Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

EVENT_LABELS = [
    "Job Interviews", "Birthday", "Graduations", "MET Gala", "Business Meeting",
    "Beach", "Picnic", "Summer", "Funeral", "Romantic Dinner",
    "Cold", "Casual", "Wedding"
]

def create_blank_tensor():
    blank_image = PILImage.new("RGB", (224, 224), (255, 255, 255))
    tensor_img = transform(blank_image).unsqueeze(0).to(device)
    return tensor_img, blank_image


from datetime import datetime, timedelta

def generate_recommendations(user_id):
    """
    Generate, predict, and save outfit combinations based on the user's newly uploaded images only.
    """
    print(f"🔄 Generating outfit combinations for user: {user_id}")

    user_images = ImageModel.query.filter_by(user_id=user_id).all()

    recent_cutoff = datetime.utcnow() - timedelta(minutes=5)
    new_images = ImageModel.query.filter_by(user_id=user_id).filter(ImageModel.created_at >= recent_cutoff).all()
    new_image_paths = {img.image_path for img in new_images}

    if not new_image_paths:
        print("⚠️ No new images found. Skipping generation.")
        return []

    category_mapping = {}
    for img in user_images:
        category_mapping.setdefault(img.category, []).append(img.image_path)

    tops = category_mapping.get("Tops", [])
    bottoms = category_mapping.get("Bottoms", [])
    shoes = category_mapping.get("Shoes", [])
    allwear = category_mapping.get("All-wear", [])
    optional_categories = {k: v for k, v in category_mapping.items() if k not in ["Tops", "Bottoms", "Shoes", "All-wear"]}
    optional_values = list(optional_categories.values())

    valid_combinations = []
    for r in range(2, 8):
        if tops and bottoms and shoes:
            base = [tops, bottoms, shoes]
            n = r - len(base)
            if n == 0:
                valid_combinations += list(product(*base))
            elif n > 0:
                for opt_comb in combinations(optional_values, n):
                    valid_combinations += list(product(*(base + list(opt_comb))))
        if allwear and shoes:
            base = [allwear, shoes]
            n = r - len(base)
            if n == 0:
                valid_combinations += list(product(*base))
            elif n > 0:
                for opt_comb in combinations(optional_values, n):
                    valid_combinations += list(product(*(base + list(opt_comb))))

    if not valid_combinations:
        print("⚠️ No valid combinations found")
        return []

    allwear_set = set(category_mapping.get("All-wear", []))
    outerwear_set = set(category_mapping.get("Outerwear", []))

    filtered_combinations = [
        combo for combo in valid_combinations
        if not (any(i in allwear_set for i in combo) and any(i in outerwear_set for i in combo))
    ]

    filtered_combinations = [
        combo for combo in filtered_combinations
        if any(path in new_image_paths for path in combo)
    ]

    if not filtered_combinations:
        print("⚠️ No new combinations with recently uploaded images")
        return []

    existing_outfits = {
        tuple(json.loads(row.outfit))
        for row in GeneratedOutfit.query.filter_by(user_id=user_id).all()
    }

    filtered_combinations = [
        combo for combo in filtered_combinations
        if tuple(combo) not in existing_outfits
    ]

    if not filtered_combinations:
        print("✅ All new combinations already exist. No duplicates generated.")
        return []

    print(f"\n🧥 {len(filtered_combinations)} NEW outfit combinations to process:\n")

    for idx, combo in enumerate(tqdm(filtered_combinations, desc="🧠 Predicting outfits"), start=1):
        outfit_files = list(combo)
        category_filename_pairs = []
        for item in outfit_files:
            cat_found = next((c for c, items in category_mapping.items() if item in items), None)
            if cat_found:
                mapped_cat = "All-body/Tops" if cat_found in ["All-body", "Tops"] else cat_found
                category_filename_pairs.append((mapped_cat, item))

        gen = GeneratedOutfit(user_id=user_id, outfit=json.dumps(outfit_files))
        db.session.add(gen)

        slot_order = ["Hats", "Accessories", "Sunglasses", "Outerwear", "All-body/Tops", "Bottoms", "Shoes"]
        CATEGORY_RENAME = {"All-body": "All-body/Tops", "Tops": "All-body/Tops", "All-wear": "All-body/Tops"}

        category_slot_mapping = {}
        for cat, fname in category_filename_pairs:
            category_slot_mapping[CATEGORY_RENAME.get(cat, cat)] = fname

        input_batch = []
        for slot in slot_order:
            if slot in category_slot_mapping:
                blob_name = category_slot_mapping[slot]
                blob = gcs_bucket.blob(blob_name)
                data = blob.download_as_bytes()
                img = PILImage.open(BytesIO(data)).convert("RGB")
                tensor_img = transform(img).unsqueeze(0).to(device)
                input_batch.append(tensor_img)
            else:
                blank_tensor, _ = create_blank_tensor()
                input_batch.append(blank_tensor)

        with torch.no_grad():
            logits, *_ = model(*input_batch)
            probs = torch.softmax(logits[0], dim=0).cpu().numpy()

        scores_dict = {EVENT_LABELS[i]: float(probs[i]) for i in range(len(EVENT_LABELS))}
        top_idx = int(probs.argmax())
        top_event, top_score = EVENT_LABELS[top_idx], float(probs[top_idx])

        res = RecommendationResult(
            user_id=user_id,
            event=top_event,
            outfit=json.dumps(outfit_files),
            scores=json.dumps(scores_dict),
            match_score=top_score,
            heatmap_paths="[]"
        )
        db.session.add(res)

    db.session.commit()
    print(f"✅ Saved {len(filtered_combinations)} new outfit predictions for user {user_id}.")
    return filtered_combinations
