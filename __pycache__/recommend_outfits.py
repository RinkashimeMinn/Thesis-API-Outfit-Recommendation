import os
import json
import torch
import gdown
from itertools import product
from PIL import Image
from torchvision import transforms
from flask_sqlalchemy import SQLAlchemy
from siamese_network import SiameseNetwork
from database import db, ImageModel, RecommendationResult
from tqdm import tqdm  # make sure you have tqdm installed

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

def create_blank_image_tensor():
    blank_image = Image.new("RGB", (224, 224), (255, 255, 255))
    return transform(blank_image).unsqueeze(0)

def generate_recommendations(user_id, new_image_paths: list[str] = None):
    """
    If new_image_paths is provided, only generate+save outfits that include
    at least one of those filenames. Otherwise, wipe & regenerate everything.
    """
    mode = "incremental" if new_image_paths else "full"
    print(f"🔄 [{mode}] Generating recommendations for user: {user_id}")

    # 1️⃣ fetch & bucket
    user_images = ImageModel.query.filter_by(user_id=user_id).all()
    category_mapping = {}
    for img in user_images:
        category_mapping.setdefault(img.category, []).append(img.image_path)

    tops    = category_mapping.get("Tops", [])
    allwear = category_mapping.get("All-wear", [])
    bottoms = category_mapping.get("Bottoms", [])
    shoes   = category_mapping.get("Shoes", [])
    optional_categories = {
        k: v for k, v in category_mapping.items()
        if k not in ["Tops", "Bottoms", "All-wear", "Shoes"]
    }

    # 2️⃣ build all raw combos
    valid_combinations = []
    for r in tqdm(range(2, 8), desc="Building combos", unit="size"):
        # Flow 1: Tops + Bottoms + Shoes
        if tops and bottoms and shoes:
            base = [tops, bottoms, shoes]
            n = r - len(base)
            if n >= 0:
                slots = base + list(optional_categories.values())[:n]
                if len(slots) == r:
                    valid_combinations += list(product(*slots))
        # Flow 2: All-wear + Shoes
        if allwear and shoes:
            base = [allwear, shoes]
            n = r - len(base)
            if n >= 0:
                slots = base + list(optional_categories.values())[:n]
                if len(slots) == r:
                    valid_combinations += list(product(*slots))

    if not valid_combinations:
        print("⚠️ No valid combinations found")
        return

    # 3️⃣ pick which combos to process
    if new_image_paths:
        combos_to_process = [
            combo for combo in valid_combinations
            if any(fn in combo for fn in new_image_paths)
        ]
    else:
        RecommendationResult.query.filter_by(user_id=user_id).delete()
        combos_to_process = valid_combinations

    if not combos_to_process:
        print("ℹ️ No new combinations to save")
        return

    # 3.5️⃣ filter out any combos mixing All-wear and Outerwear
    allwear_set = set(category_mapping.get("All-wear", []))
    outer_set   = set(category_mapping.get("Outerwear", []))
    combos_to_process = [
        combo for combo in combos_to_process
        if not (any(item in allwear_set for item in combo) and
                any(item in outer_set   for item in combo))
    ]
    if not combos_to_process:
        print("⚠️ All combos with All-wear+Outerwear have been removed")
        return

    print(f"🔢 Inferring {len(combos_to_process)} outfits ({mode} mode)")
    upload_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
    blank_tensor = create_blank_image_tensor()

    # 4️⃣ inference + save
    for outfit in tqdm(combos_to_process, desc="Inferring outfits", unit="outfit"):
        filled = list(outfit) + ["BLANK"] * (7 - len(outfit))
        tensors = []
        for fn in filled:
            if fn == "BLANK":
                tensors.append(blank_tensor)
            else:
                img = Image.open(os.path.join(upload_dir, fn)).convert("RGB")
                tensors.append(transform(img).unsqueeze(0).to(device))

        with torch.no_grad():
            logits, *_ = model(*tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        labels = [
            "Job Interviews","Birthday","Graduations","MET Gala","Business Meeting",
            "Beach","Picnic","Summer","Funeral","Romantic Dinner","Cold","Casual","Wedding"
        ]
        scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
        db.session.add(RecommendationResult(
            user_id=user_id,
            event="N/A",
            outfit=json.dumps(list(outfit)),
            scores=json.dumps(scores),
            match_score=float(probs.max()),
            heatmap_paths="[]"
        ))

    db.session.commit()
    print(f"✅ {mode.capitalize()} saved {len(combos_to_process)} recommendation(s) for user {user_id}")
