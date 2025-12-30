from flask import Flask, request, jsonify, redirect, abort
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import threading
import json
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from google.cloud import storage
from google.oauth2 import service_account

from database import db, ImageModel, RecommendationResult, User, Saved, UploadedImage
from recommend_outfits import generate_recommendations
from event_predictor import predict_event_from_filenames

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# ✅ SQLite3 Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.abspath("assets/database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

SERVICE_ACCOUNT_PATH = "/etc/secrets/morphfit-key"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_PATH
)
gcs_client = storage.Client(
    credentials=credentials,
    project=credentials.project_id
)
gcs_bucket = gcs_client.bucket("morphfit-thesis")

# Helper to build public URL for a blob
def gcs_url(filename):
    return f"https://storage.googleapis.com/{gcs_bucket.name}/{filename}"

bcrypt = Bcrypt(app)
db.init_app(app)
with app.app_context():
    db.create_all()

# ------------------- ROUTES -------------------

@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    blob = gcs_bucket.blob(filename)
    if not blob.exists():
        abort(404)
    # Redirect client directly to the public URL
    return redirect(blob.public_url, code=302)

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or any(k not in data for k in ("username","password","security_question","security_answer")):
        return jsonify({"error": "Missing required fields"}), 400

    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Username already exists"}), 400

    new_user = User(
        username=data["username"],
        password=bcrypt.generate_password_hash(data["password"]).decode('utf-8'),
        security_question=data["security_question"],
        security_answer=data["security_answer"]
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully!"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or any(k not in data for k in ("username","password")):
        return jsonify({"error": "Missing required fields"}), 400

    user = User.query.filter_by(username=data["username"]).first()
    if user and bcrypt.check_password_hash(user.password, data["password"]):
        return jsonify({"message": "Login successful", "user_id": user.id}), 200
    return jsonify({"error": "Invalid username or password"}), 401

@app.route('/get_user_info', methods=['GET'])
def get_user_info():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'username': user.username})

@app.route("/get_security_question", methods=["POST"])
def get_security_question():
    username = request.get_json().get("username")
    if not username:
        return jsonify({"error": "Missing username"}), 400
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"question": user.security_question}), 200

@app.route("/reset_password", methods=["POST"])
def reset_password():
    data = request.get_json()
    if any(k not in data for k in ("username","answer","new_password")):
        return jsonify({"error": "Missing fields"}), 400
    user = User.query.filter_by(username=data["username"]).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
    if user.security_answer.strip().lower() != data["answer"].strip().lower():
        return jsonify({"error": "Incorrect answer"}), 403
    user.password = bcrypt.generate_password_hash(data["new_password"]).decode('utf-8')
    db.session.commit()
    return jsonify({"message": "Password reset successfully!"}), 200

@app.route("/upload-multiple", methods=["POST"])
def upload_multiple_images():
    if "images" not in request.files or any(k not in request.form for k in ("user_id","category")):
        return jsonify({"error": "Missing required fields"}), 400

    user_id = int(request.form["user_id"])
    category = request.form["category"]
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "Invalid user ID"}), 400

    prefix = {"Tops":"TOP","Bottoms":"BTM","Shoes":"SHO","Outerwear":"OUT",
              "All-wear":"ALL","Accessories":"ACC","Hats":"HAT","Sunglasses":"SUN"}
    code = prefix.get(category, "GEN")

    # determine start index
    existing = ImageModel.query.filter_by(user_id=user_id, category=category).all()
    used_nums = [int(img.id.split(code)[-1]) for img in existing if code in img.id and img.id.split(code)[-1].isdigit()]
    start = max(used_nums, default=0) + 1

    uploaded = []
    for idx, image in enumerate(request.files.getlist("images")):
        seq = start + idx
        image_id = f"U{user_id}_{code}{seq:02d}"
        base = secure_filename(image.filename).rsplit('.',1)[0]
        fname = f"{uuid.uuid4().hex}_{base}.jpg"
        data = image.read()
        if len(data) < 1000:
            return jsonify({"error": "Uploaded image is too small or empty."}), 400

        # upload to GCS
        blob = gcs_bucket.blob(fname)
        blob.upload_from_string(data, content_type=image.mimetype)

        db.session.add(ImageModel(id=image_id, image_path=fname, category=category, user_id=user_id))
        uploaded.append({"image_id": image_id, "image_path": blob.public_url})

    db.session.commit()
    
    def run_with_context(uid):
        with app.app_context():
            generate_recommendations(uid)

    threading.Thread(target=run_with_context, args=(user_id,)).start()


    return jsonify({"message": "Images uploaded successfully! Recommendations are being generated.",
                    "images": uploaded}), 201

@app.route("/delete-images", methods=["POST"])
def delete_images():
    ids = request.get_json().get("image_ids", [])
    if not ids:
        return jsonify({"error": "No images selected"}), 400

    images = ImageModel.query.filter(ImageModel.id.in_(ids)).all()
    recs_del = sum(1 for rec in RecommendationResult.query.all()
                   if any(fn in json.loads(rec.outfit) for fn in [img.image_path for img in images]))
    for rec in RecommendationResult.query.all():
        if any(fn in json.loads(rec.outfit) for fn in [img.image_path for img in images]):
            db.session.delete(rec)

    saved_del = 0
    for s in Saved.query.all():
        clothes = s.clothes_ids if isinstance(s.clothes_ids, list) else json.loads(s.clothes_ids)
        if any(i in ids for i in clothes):
            db.session.delete(s)
            saved_del += 1

    files_del = 0
    for img in images:
        blob = gcs_bucket.blob(img.image_path)
        try:
            blob.delete()
            files_del += 1
        except:
            pass
        db.session.delete(img)

    db.session.commit()
    return jsonify({"message": f"Deleted {len(images)} image record(s), removed {saved_del} saved outfit(s), {recs_del} recommendation(s), and deleted {files_del} file(s)."}), 200

@app.route("/delete-all/<category>", methods=["DELETE"])
def delete_all_images(category):
    images = ImageModel.query.filter_by(category=category).all()
    if not images:
        return jsonify({"message": f"No images found in '{category}'."}), 200

    recs_del = sum(1 for rec in RecommendationResult.query.all()
                   if any(fn in json.loads(rec.outfit) for fn in [img.image_path for img in images]))
    for rec in RecommendationResult.query.all():
        if any(fn in json.loads(rec.outfit) for fn in [img.image_path for img in images]):
            db.session.delete(rec)

    saved_del = 0
    for s in Saved.query.all():
        clothes = s.clothes_ids if isinstance(s.clothes_ids, list) else json.loads(s.clothes_ids)
        if any(i == img.id for img in images for i in clothes):
            db.session.delete(s)
            saved_del += 1

    files_del = 0
    for img in images:
        blob = gcs_bucket.blob(img.image_path)
        try:
            blob.delete()
            files_del += 1
        except:
            pass
        db.session.delete(img)

    db.session.commit()
    return jsonify({"message": f"All {len(images)} image record(s) in '{category}' deleted, {saved_del} saved outfit(s), {recs_del} recommendation(s) removed, and {files_del} file(s) cleaned up."}), 200

@app.route("/images/<category>", methods=["GET"])
def get_images_by_category(category):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    images = ImageModel.query.filter_by(category=category, user_id=user_id).all()
    return jsonify([{"id": img.id, "image_path": gcs_url(img.image_path), "category": img.category} for img in images]), 200

@app.route("/images/user/<user_id>", methods=["GET"])
def get_user_images(user_id):
    imgs = ImageModel.query.filter_by(user_id=user_id).all()
    return jsonify([{"id": img.id, "image_path": gcs_url(img.image_path), "category": img.category} for img in imgs]), 200

@app.route("/recommend", methods=["POST"])
def recommend_outfit():
    try:
        data = request.get_json(); event = data.get("event"); user_id = data.get("user_id")
        if not event or not user_id:
            return jsonify({"error": "Missing event or user ID"}), 400
        threshold = 0.20
        # DL-based
        recs = []
        for rec in RecommendationResult.query.filter_by(user_id=user_id):
            scores = json.loads(rec.scores)
            sc = scores.get(event, 0)
            if sc >= threshold:
                fns = json.loads(rec.outfit)
                recs.append({"match_score": sc,
                             "outfit": [gcs_url(fn) for fn in fns],
                             "raw_filenames": fns,
                             "scores": scores})
        # FP-Growth
        transactions, lookup = [], {}
        for s in Saved.query.filter_by(user_id=user_id):
            files = [os.path.basename(ImageModel.query.get(i).image_path) for i in s.clothes_ids]
            if files:
                transactions.append(files)
                lookup[tuple(sorted(files))] = s.event.lower()
        freq_event = []
        if transactions:
            te = TransactionEncoder(); df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
            fi = fpgrowth(df, min_support=0.3, use_colnames=True)
            rules = association_rules(fi, metric="confidence", min_threshold=0.5)
            for _,r in rules.iterrows():
                itemset = sorted(r['antecedents'].union(r['consequents']))
                for trans,evt in lookup.items():
                    if set(itemset).issubset(set(trans)) and evt==event.lower():
                        freq_event.append({"match_score":0,"boost_score":len(itemset),
                                           "outfit":[gcs_url(it) for it in itemset],
                                           "raw_filenames":itemset,
                                           "support":r['support'],"confidence":r['confidence'],"lift":r['lift']})
                        break
        # boost
        def boost(fns): return sum(len(fp['raw_filenames']) for fp in freq_event if set(fp['raw_filenames']).issubset(set(fns)))
        for o in recs: o['boost_score']=boost(o['raw_filenames'])
        combined = sorted(recs+freq_event, key=lambda x:(x['boost_score'],x.get('match_score',0)), reverse=True)
        return jsonify({"event": event, "results": combined}), 200
    except Exception as e:
        print(f"❌ Recommend API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/classify_event', methods=['POST'])
def classify_event():
    files = request.files.getlist('images')
    cats = request.form.getlist('categories')
    if len(files)!=len(cats): return jsonify({'error':'Mismatch'}),400
    saved=[]
    for cat,file in zip(cats,files):
        ext=file.filename.rsplit('.',1)[1].lower()
        name=f"{cat.replace('/','-')}_{uuid.uuid4().hex[:8]}.{ext}"
        blob=gcs_bucket.blob(name)
        blob.upload_from_string(file.read(), content_type=file.mimetype)
        db.session.add(UploadedImage(category=cat, filename=name))
        saved.append((cat,name))
    db.session.commit()
    try:
        return jsonify(predict_event_from_filenames(saved))
    except Exception as e:
        print(e)
        return jsonify({'error':str(e)}),500

@app.route('/save_outfit', methods=['POST'])
def save_outfit():
    data = request.get_json()
    if any(k not in data for k in ('user_id','event','outfit')):
        return jsonify({'error':'Missing data'}),400
    ids=[]
    for url in data['outfit']:
        fname=url.split('/')[-1]
        rec=ImageModel.query.filter(ImageModel.image_path.like(f"%{fname}"), ImageModel.user_id==data['user_id']).first()
        if rec: ids.append(rec.id)
    s=Saved(user_id=data['user_id'], event=data['event'], outfit=data['outfit'], clothes_ids=ids)
    db.session.add(s); db.session.commit()
    return jsonify({'message':'Outfit saved successfully!','image_ids':ids}),201

@app.route('/remove_outfit_by_id', methods=['POST'])
def remove_outfit_by_id():
    oid=request.get_json().get('id')
    if not oid: return jsonify({'error':'Missing outfit ID'}),400
    outfit=Saved.query.get(oid)
    if not outfit: return jsonify({'error':'Outfit not found'}),404
    db.session.delete(outfit); db.session.commit()
    return jsonify({'message':'Outfit removed successfully'}),200

@app.route('/fp_growth_saved', methods=['GET'])
def fp_growth_saved():
    user_id=request.args.get('user_id')
    if not user_id: return jsonify({'error':'Missing user_id'}),400
    saved=Saved.query.filter_by(user_id=user_id).all()
    transactions=[]; details={}
    for s in saved:
        ids=[]
        for img_id in s.clothes_ids:
            img=ImageModel.query.get(img_id)
            if img:
                ids.append(img.id)
                details[img.id]={'id':img.id,'image_path':gcs_url(img.image_path)}
        if ids: transactions.append(ids)
    if not transactions: return jsonify({'error':'No transactions'}),404
    te=TransactionEncoder(); df=pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
    fi=fpgrowth(df, min_support=0.3, use_colnames=True)
    rules=association_rules(fi, metric='confidence', min_threshold=0.5)
    result=[]
    for _,r in rules.iterrows():
        comb=r['antecedents'].union(r['consequents'])
        items=[details[i] for i in comb if i in details]
        result.append({'itemsets':items,'support':r['support'],'confidence':r['confidence'],'lift':r['lift']})
    return jsonify({'frequent_itemsets': result}),200

@app.route('/get_saved_outfits', methods=['GET'])
def get_saved_outfits():
    user_id=request.args.get('user_id')
    if not user_id: return jsonify({'error':'Missing user_id'}),400
    saved=Saved.query.filter_by(user_id=user_id).all()
    data=[]
    for s in saved:
        outfit=json.loads(s.outfit) if not isinstance(s.outfit,list) else s.outfit
        data.append({'id':s.id,'event':s.event,'outfit':outfit,'clothes_ids':s.clothes_ids})
    return jsonify({'saved_outfits':data}),200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

