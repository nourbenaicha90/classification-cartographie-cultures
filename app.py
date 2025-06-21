import os
import secrets
import tifffile
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import matplotlib.colors as colors
import rasterio
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import base64

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Create necessary folders if not exist



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


ALLOWED_EXTENSIONS = {'tif', 'tiff'}

model = xgb.XGBClassifier()
model.load_model("xgb_model.h5")

known_classes = ['non-agricultural', 'agricultural']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(img):
    features = np.mean(img, axis=(0, 1)).astype(np.float32).reshape(1, -1)
    preds = model.predict_proba(features)
    class_idx = int(np.argmax(preds))
    confidence = preds[0][class_idx] * 100
    predicted_class = known_classes[class_idx]
    return f"{predicted_class} ({confidence:.2f}%)"

def save_natural_rgb_image(img, save_path):
    red = img[:, :, 3]
    green = img[:, :, 2]
    blue = img[:, :, 1]
    def normalize(b):
        return (b - b.min()) / (b.max() - b.min() + 1e-6)
    rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])
    plt.imsave(save_path, rgb)

def calculate_green_percentage(img):
    red = img[:, :, 3].astype(np.float32)
    nir = img[:, :, 7].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    green_pixels = (ndvi > 0.2).sum()
    total_pixels = ndvi.size
    return (green_pixels / total_pixels) * 100

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     return render_template('logout.html')

# @app.route('/create_account')
# def create_account():
#     return render_template('create_account.html')

@app.route('/newpage')
def newpage():
    return render_template('newpage.html')



@app.route('/analyse', methods=['GET', 'POST'])
def analyse_page():
    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    def save_image_with_colorbar(data, output_path, title):
        fig, ax = plt.subplots(figsize=(5, 4))
        cmap = plt.cm.RdYlGn
        norm = colors.Normalize(vmin=0, vmax=1)
        img = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.axis('off')
        cbar = fig.colorbar(img, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label('Index Value', rotation=270, labelpad=15)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def process_tif(filepath, output_folder):
        with rasterio.open(filepath) as src:
            img = src.read()
            red = img[2].astype(float)
            nir = img[3].astype(float)
            blue = img[0].astype(float)
            green = img[1].astype(float)

            ndvi = (nir - red) / (nir + red + 1e-6)
            savi = ((nir - red) / (nir + red + 0.5)) * 1.5
            evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
            ndwi = (green - nir) / (green + nir + 1e-6)
            swir1 = img[4].astype(float) if img.shape[0] > 4 else np.zeros_like(nir)
            lswi = (nir - swir1) / (nir + swir1 + 1e-6)

            images = {}
            base = os.path.splitext(os.path.basename(filepath))[0]

            for name, data in [('NDVI', ndvi), ('SAVI', savi), ('EVI', evi), ('NDWI', ndwi), ('LSWI', lswi)]:
                norm_data = normalize(data)
                output_file = f'{base}_{name}.png'
                output_path = os.path.join(IMAGE_FOLDER, output_file)
                save_image_with_colorbar(norm_data, output_path, name)
                images[name] = output_file

            rgb = np.dstack([normalize(img[2]), normalize(img[1]), normalize(img[0])])
            natural_path = os.path.join(IMAGE_FOLDER, f'{base}_natural.png')
            plt.imsave(natural_path, rgb)
            # images['Natural'] = f'{base}_natural.png'

            mean_ndvi = np.nanmean(ndvi)
            mean_savi = np.nanmean(savi)
            mean_evi = np.nanmean(evi)
            mean_ndwi = np.nanmean(ndwi)
            mean_lswi = np.nanmean(lswi)


            # تحليل صلاحية التربة
            if mean_ndvi > 0.2 and mean_savi > 0.3 and mean_evi > 0.3 and mean_ndwi > 0 and mean_lswi > 0:
                status = "✅ The soil is suitable for agriculture"
            else:
                status = "❌ The soil is not suitable or needs improvement"

            explanation = {
                'NDVI': f"NDVI = {mean_ndvi:.2f} → {'Good vegetation cover' if mean_ndvi > 0.4 else 'Poor vegetation'}",
                'SAVI': f"SAVI = {mean_savi:.2f} → {'Good soil influence reduction' if mean_savi > 0.3 else 'Low performance'}",
                'EVI': f"EVI = {mean_evi:.2f} → {'Healthy plant growth' if mean_evi > 0.3 else 'Weak growth'}",
                'NDWI': f"NDWI = {mean_ndwi:.2f} → {'Good moisture' if mean_ndwi > 0 else 'Dry soil'}",
                'LSWI': f"LSWI = {mean_lswi:.2f} → {'Good water content' if mean_lswi > 0 else 'Low water content'}"

            }


            images['Status'] = status
            images['Explanation'] = explanation
            images['Natural'] = f'{base}_natural.png'  # ✅ CORRECT



            return images

    images = {}
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            try:
                images = process_tif(filepath, IMAGE_FOLDER)
            except Exception as e:
                flash(f"❌ خطأ أثناء تحليل الصورة: {str(e)}")
                return redirect(request.url)
        else:
            flash("📁 يرجى اختيار ملف TIF صالح")
            return redirect(request.url)

    return render_template('Analyse.html', images=images)


@app.route('/analyse_result')
def analyse_result_page():
    return render_template('analyse_result.html')

@app.route('/important', methods=["GET", "POST"])
def important_class():
    result = None
    img_info = {}
    rgb_image_path = None
    green_percent = None
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("📁 يرجى اختيار ملف .tif")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("❌ نوع الملف غير مدعوم. يرجى رفع ملف بصيغة .tif أو .tiff فقط.")
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            img = tifffile.imread(filepath)
        except Exception as e:
            flash(f"❌ خطأ في تحميل الصورة: {str(e)}")
            return redirect(request.url)
        if img.ndim != 3:
            flash("❌ الصورة يجب أن تكون ثلاثية الأبعاد (ارتفاع، عرض، عدد القنوات).")
            return redirect(request.url)
        result = classify_image(img)
        green_percent = calculate_green_percentage(img)
        img_info['height'], img_info['width'], img_info['bands'] = img.shape
        rgb_filename = filename.rsplit('.', 1)[0] + "_natural.png"
        rgb_save_path = os.path.join(IMAGE_FOLDER, rgb_filename)
        save_natural_rgb_image(img, rgb_save_path)
        rgb_image_path = f"images/{rgb_filename}"
    return render_template("important_class.html",
                           result=result,
                           img_info=img_info,
                           rgb_image=rgb_image_path,
                           green_percent=green_percent)

@app.route('/segment', methods=['GET', 'POST'])
def segment():
    if request.method == 'GET':
        return render_template('segment.html')

    if 'image' not in request.files:
        return render_template('segment.html', error='📁 لم يتم تحميل أي صورة')

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception:
        return render_template('segment.html', error='❌ ملف الصورة غير صالح')

    image_np = np.array(img)
    image_resized = cv2.resize(image_np, (256, 256))
    img_flat = image_resized.reshape((-1, 3))

    kmeans = KMeans(n_clusters=2, random_state=0).fit(img_flat)
    labels = kmeans.labels_.reshape(image_resized.shape[:2])
    mask = (labels == labels[0, 0])

    # Count green pixels
    green_pixels = np.sum(mask)
    total_pixels = mask.size
    green_percentage = round((green_pixels / total_pixels) * 100, 2)  # Rounded to 2 decimal places

    segmented = np.zeros_like(image_resized)
    segmented[mask] = [0, 255, 0]       # Green
    segmented[~mask] = [255, 0, 0]      # Red
    segmented = cv2.resize(segmented, (image_np.shape[1], image_np.shape[0]))

    _, buffer = cv2.imencode('.png', segmented)
    encoded = base64.b64encode(buffer).decode('utf-8')

    return render_template('segment.html', result=encoded, percentage=green_percentage)

@app.route('/classify-crop', methods=['GET', 'POST'])
def classify_crop():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file and image_file.filename.endswith(('.tif', '.tiff')):
            filename = secure_filename(image_file.filename)
            input_path = os.path.join('uploads', filename)
            image_file.save(input_path)

            with rasterio.open(input_path) as src:
                bands = src.read([3, 4, 5, 6, 8, 11]).astype(np.float32)
                B3, B4, B5, B6, B8, B11 = bands

                NDVI = (B8 - B4) / (B8 + B4 + 1e-6)
                GNDVI = (B8 - B3) / (B8 + B3 + 1e-6)
                NDRE = (B5 - B4) / (B5 + B4 + 1e-6)
                NDWI = (B3 - B8) / (B3 + B8 + 1e-6)

                classified = np.zeros(NDVI.shape, dtype=np.uint8)

                # خضر
                green_mask = (NDVI >= 0.6) & (NDVI <= 0.9) & (GNDVI > 0.5) & (NDWI > 0.3)
                classified[green_mask] = 1

                # قمح
                wheat_mask = (NDVI >= 0.5) & (NDVI <= 0.75) & (NDRE >= 0.2) & (NDRE <= 0.4) & (NDWI < 0.2)
                classified[wheat_mask] = 2

                # شعير
                barley_mask = (NDVI >= 0.4) & (NDVI <= 0.7) & (NDRE >= 0.15) & (NDRE <= 0.35) & (NDWI < 0.2)
                classified[barley_mask] = 3

                # ألوان المحاصيل
                color_map = {
                    0: [0, 0, 0],       # غير معروف
                    1: [0, 255, 0],     # خضر
                    2: [255, 255, 0],   # قمح
                    3: [255, 0, 0],     # شعير
                }

                # أسماء المحاصيل
                crop_labels = {
                    0: "Unknown",
                    1: "Vegetables",
                    2: "Wheat",
                    3: "Barley"
                }

                # إنشاء صورة RGB
                h, w = classified.shape
                rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
                for val, color in color_map.items():
                    rgb_image[classified == val] = color

                # حفظ الصورة
                output_filename = f"classified_{filename.rsplit('.', 1)[0]}.png"
                output_path = os.path.join('static', 'images', output_filename)
                cv2.imwrite(output_path, rgb_image)

                # حساب نسبة كل محصول
                total_pixels = h * w
                unique, counts = np.unique(classified, return_counts=True)
                percentages = {crop_labels[k]: round((counts[i] / total_pixels) * 100, 2)
                               for i, k in enumerate(unique)}

                # تحديد المحصول المسيطر
                main_crop_code = unique[np.argmax(counts)]
                main_crop_name = crop_labels[main_crop_code]

                return render_template("classify_crop.html",
                                       classification_map=f"images/{output_filename}",
                                       percentages=percentages,
                                       main_crop=main_crop_name)

    return render_template("classify_crop.html", classification_map=None)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)