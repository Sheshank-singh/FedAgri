import os
import io
import json
import time
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Optional, Tuple

from models.model_factory import create_model
from data.dataset import get_transforms


# Cache loaded models in memory: checkpoint_path -> nn.Module
MODEL_CACHE: Dict[str, torch.nn.Module] = {}


# Agronomic advisory knowledge base for PlantVillage classes
REMEDIES: Dict[str, Dict[str, str]] = {
    "Apple___Apple_scab": {
        "severity": "Moderate to High",
        "description": "Fungal disease caused by Venturia inaequalis producing olive-green to black velvety spots on leaves.",
        "remedy": "Apply preventative fungicides like captan, mancozeb, or copper soaps early in spring. Rake and destroy fallen leaves to reduce overwintering spore inoculum."
    },
    "Apple___Black_rot": {
        "severity": "Severe",
        "description": "Caused by Diplodia seriata (Botryosphaeria obtusa); induces leaf 'frog-eye' leaf spots, cankers, and fruit rot.",
        "remedy": "Prune out dead wood and mummified fruit during winter. Apply captan or thiophanate-methyl fungicides from pink bud stage through harvest."
    },
    "Apple___Cedar_apple_rust": {
        "severity": "Moderate",
        "description": "Gymnosporangium juniperi-virginianae rust requires alternate hosts (Juniper/Red Cedar) and causes bright orange-yellow leaf spots.",
        "remedy": "Remove nearby eastern red cedar trees if feasible within 1-2 miles. Apply myclobutanil or mancozeb at bud break."
    },
    "Apple___healthy": {
        "severity": "None",
        "description": "Vibrant, healthy foliage free of fungal or bacterial lesions.",
        "remedy": "Maintain regular pruning, balanced nitrogen fertilization, and good soil drainage."
    },
    "Blueberry___healthy": {
        "severity": "None",
        "description": "Healthy blueberry foliage with uniform green pigmentation.",
        "remedy": "Keep soil pH acidic (4.5–5.5), maintain pine bark or sawdust mulch, and monitor soil moisture."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "Moderate",
        "description": "Podosphaera clandestina causes white powdery fungal growth on leaves and twigs.",
        "remedy": "Apply sulfur or myclobutanil when leaves first emerge. Ensure open canopy pruning for air circulation."
    },
    "Cherry_(including_sour)___healthy": {
        "severity": "None",
        "description": "Healthy cherry foliage with no mildew or shot-hole lesions.",
        "remedy": "Maintain annual canopy pruning, adequate irrigation, and dormant copper spray."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "Severe",
        "description": "Cercospora zeae-maydis causes rectangular, tan-to-gray lesions delimited by leaf veins.",
        "remedy": "Plant resistant hybrid seed varieties, practice crop rotation with non-host crops, and apply foliar strobilurin or triazole fungicides at tasseling."
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Moderate",
        "description": "Puccinia sorghi causes oval to elongate golden-brown pustules on both leaf surfaces.",
        "remedy": "Plant rust-resistant hybrids. If infection strikes before silking and pustules cover >5% of ear leaf area, apply azoxystrobin/propiconazole."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "Severe",
        "description": "Exserohilum turcicum produces cigar-shaped, grayish-green to tan lesions up to 6 inches long.",
        "remedy": "Use resistant maize hybrids, deep tillage of crop debris, and timely fungicide applications (pyraclostrobin or metconazole) around tassel emergence."
    },
    "Corn_(maize)___healthy": {
        "severity": "None",
        "description": "Robust, deep green corn leaves with vigorous photosynthetic tissue.",
        "remedy": "Ensure proper nitrogen side-dressing and maintain weed control between rows."
    },
    "Grape___Black_rot": {
        "severity": "Severe",
        "description": "Guignardia bidwellii attacks all green parts, producing circular reddish-brown leaf spots with black pycnidia rings.",
        "remedy": "Apply mancozeb, captan, or kresoxim-methyl starting at 1-inch shoot growth. Prune vines to improve sun exposure and air flow."
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "Severe",
        "description": "Fungal trunk complex causing 'tiger-stripe' interveinal chlorosis/necrosis and spotted berries.",
        "remedy": "Prune in dry weather; paint pruning wounds with wound sealant or biocontrol agents (Trichoderma). Remove severely infected vines."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "Moderate",
        "description": "Pseudocercospora cladosporioides causes well-defined brown spots with dark borders.",
        "remedy": "Post-harvest copper fungicide spray. Improve canopy management to speed drying after rainfall."
    },
    "Grape___healthy": {
        "severity": "None",
        "description": "Healthy grapevine leaf showing pristine palmate structure and clear venation.",
        "remedy": "Practice balanced shoot positioning, shoot thinning, and regulated deficit irrigation."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "Critical",
        "description": "Bacterial infection by Candidatus Liberibacter asiaticus vectored by Asian citrus psyllid; causes blotchy mottle chlorosis.",
        "remedy": "Strict vector control using systemic insecticides (imidacloprid), nutritional foliar sprays to support root health, and prompt rogueing of infected trees."
    },
    "Peach___Bacterial_spot": {
        "severity": "High",
        "description": "Xanthomonas arboricola pv. pruni causes water-soaked angular leaf lesions that drop out, creating a 'shot-hole' appearance.",
        "remedy": "Apply oxytetracycline or low-rate fixed copper bactericides from shuck-split through petal fall. Plant less susceptible cultivars."
    },
    "Peach___healthy": {
        "severity": "None",
        "description": "Vibrant, intact peach leaf with no shot-holes or chlorosis.",
        "remedy": "Maintain open-center tree training, apply dormant copper spray, and monitor peach tree borer."
    },
    "Pepper,_bell___Bacterial_spot": {
        "severity": "High",
        "description": "Xanthomonas euvesicatoria causes small, circular to irregular water-soaked spots turning dark brown.",
        "remedy": "Use certified pathogen-free seed and resistant hybrids. Apply copper hydroxide mixed with mancozeb, and avoid overhead sprinkler irrigation."
    },
    "Pepper,_bell___healthy": {
        "severity": "None",
        "description": "Healthy bell pepper leaf showing vigorous cellular structure and deep green color.",
        "remedy": "Mulch soil with plastic or straw to conserve moisture and prevent soil-splash pathogens."
    },
    "Potato___Early_blight": {
        "severity": "Moderate to High",
        "description": "Alternaria solani causes dark brown to black target-board lesions with concentric rings.",
        "remedy": "Apply protectant fungicides (chlorothalonil or mancozeb) on 7-10 day schedules. Ensure adequate nitrogen and potassium fertility to avoid plant stress."
    },
    "Potato___Late_blight": {
        "severity": "Critical",
        "description": "Phytophthora infestans (water mold) causes rapid water-soaked lesions with white sporulation on leaf undersides.",
        "remedy": "Immediate intervention required: apply systemic fungicides such as mandipropamid, cyazofamid, or fluopicolide. Destroy volunteer potatoes and infected tubers."
    },
    "Potato___healthy": {
        "severity": "None",
        "description": "Healthy potato leaf with clean margins and optimal turgidity.",
        "remedy": "Hill soil properly around developing tubers and practice 3-year crop rotation."
    },
    "Raspberry___healthy": {
        "severity": "None",
        "description": "Pristine raspberry foliage without cane blight, anthracnose, or rust.",
        "remedy": "Prune spent floricanes after fruiting, maintain trellis support, and mulch root zones."
    },
    "Soybean___healthy": {
        "severity": "None",
        "description": "Uniform trifoliate soybean leaves with zero frogeye leaf spots or rust.",
        "remedy": "Maintain balanced fertility (P and K), ensure proper seeding density, and scout weekly."
    },
    "Squash___Powdery_mildew": {
        "severity": "Moderate",
        "description": "Podosphaera xanthii coats leaf surfaces with powdery white talcum-like mycelial colonies.",
        "remedy": "Apply potassium bicarbonate, neem oil, or azoxystrobin at first symptom appearance. Water at base to keep leaves dry."
    },
    "Strawberry___Leaf_scorch": {
        "severity": "Moderate",
        "description": "Diplocarpon earlianum causes small, irregular purple blotches that enlarge and turn brown.",
        "remedy": "Remove dead leaves after harvest. Apply captan or fenhexamid at bloom, and use drip irrigation instead of overhead sprinklers."
    },
    "Strawberry___healthy": {
        "severity": "None",
        "description": "Crisp, healthy trifoliate strawberry leaf showing vigorous growth.",
        "remedy": "Renovate beds annually, mulch with clean straw, and replace planting stock every 3 years."
    },
    "Tomato___Bacterial_spot": {
        "severity": "High",
        "description": "Xanthomonas perforans causes small, dark, water-soaked circular spots with yellow halos.",
        "remedy": "Apply copper + mancozeb tank mix or Actigard systemic SAR inducer. Disinfect stakes and avoid working in wet foliage."
    },
    "Tomato___Early_blight": {
        "severity": "High",
        "description": "Alternaria linariae causes brown to black concentric bullseye rings surrounded by chlorotic yellow halos on lower leaves.",
        "remedy": "Remove lower infected foliage. Apply chlorothalonil or copper soap preventatively. Stake tomatoes off the ground and water at soil level."
    },
    "Tomato___Late_blight": {
        "severity": "Critical",
        "description": "Phytophthora infestans causes large, dark water-soaked lesions that rapidly collapse the plant canopy under cool, wet weather.",
        "remedy": "Immediately apply systemic fungicides (e.g. cymoxanil, mandipropamid). Rogue heavily blighted plants to protect neighboring crops."
    },
    "Tomato___Leaf_Mold": {
        "severity": "Moderate",
        "description": "Passalora fulva causes pale green or yellowish spots on leaf tops with velvety olive-brown spores underneath.",
        "remedy": "Crucial to reduce greenhouse/tunnel humidity below 85% and maximize ventilation. Apply chlorothalonil or copper fungicides."
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "High",
        "description": "Septoria lycopersici causes abundant tiny circular spots with dark brown margins and sunken grayish centers with black speck pycnidia.",
        "remedy": "Mulch generously to block soil-splashed fungal spores. Spray chlorothalonil or mancozeb at 7-day intervals starting after first flower set."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "High",
        "description": "Tetranychus urticae causes fine yellow stippling and bronzing of leaves, often accompanied by silky webbing.",
        "remedy": "Apply insecticidal soap, horticultural oils, or bifenazate miticide. Introduce beneficial predatory mites (Phytoseiulus persimilis)."
    },
    "Tomato___Target_Spot": {
        "severity": "High",
        "description": "Corynespora cassiicola produces brown circular lesions with concentric target rings that rapidly defoliate plants.",
        "remedy": "Improve row spacing for airflow, remove crop debris, and spray azoxystrobin or difenoconazole."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "Critical",
        "description": "TYLCV transmitted by silverleaf whiteflies (Bemisia tabaci) causes severe upward leaf cupping, stunted bushy growth, and flower abortion.",
        "remedy": "Control whiteflies using yellow sticky traps and insecticidal soaps/spirotetramat. Use fine insect exclusion netting and rogue infected plants immediately."
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "Severe",
        "description": "Tobamovirus causing mottled light/dark green mosaic patterns, leaf blistering, and distorted shoe-string leaves.",
        "remedy": "Seed sanitation; wash hands and tools with 10% trisodium phosphate (TSP) or skim milk before handling plants. Destroy infected plants."
    },
    "Tomato___healthy": {
        "severity": "None",
        "description": "Robust, deeply lobed tomato foliage with strong turgor pressure and uniform emerald green hue.",
        "remedy": "Maintain regular watering to prevent blossom end rot, prune suckers for optimal airflow, and feed with balanced calcium-rich tomato fertilizer."
    }
}


def load_classes_mapping(config_path: str = "configs/config.yaml") -> Tuple[Dict[str, int], Dict[int, str]]:
    classes_json_path = "data/processed/classes.json"
    if os.path.exists(classes_json_path):
        with open(classes_json_path, "r") as f:
            data = json.load(f)
            class_to_idx = data["class_to_idx"]
            idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
            return class_to_idx, idx_to_class

    # Fallback to scanning raw directory
    from data.dataset import get_class_names
    class_names = get_class_names(config_path)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    return class_to_idx, idx_to_class


def infer_model_name_from_checkpoint(checkpoint_filename: str) -> str:
    fname = os.path.basename(checkpoint_filename).lower()
    if "mobilenetv3" in fname:
        return "mobilenetv3"
    elif "resnet50" in fname:
        return "resnet50"
    elif "efficientnet_b0" in fname:
        return "efficientnet_b0"
    elif "convnext_tiny" in fname:
        return "convnext_tiny"
    return "mobilenetv3"


def get_or_load_model(checkpoint_path: str, model_name: Optional[str] = None, device: Optional[torch.device] = None) -> torch.nn.Module:
    global MODEL_CACHE
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = f"{checkpoint_path}_{device.type}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    if not model_name:
        model_name = infer_model_name_from_checkpoint(checkpoint_path)

    _, idx_to_class = load_classes_mapping()
    num_classes = len(idx_to_class)

    print(f"[Model Loader] Instantiating {model_name} (num_classes={num_classes}) on {device}...")
    model = create_model(model_name, num_classes=num_classes, pretrained=False)

    if os.path.exists(checkpoint_path):
        print(f"[Model Loader] Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"[Model Loader] Warning: Checkpoint {checkpoint_path} not found. Using initialized model weights.")

    model.to(device)
    model.eval()
    MODEL_CACHE[cache_key] = model
    return model


def predict_leaf_disease(
    image_bytes: bytes,
    checkpoint_name: str,
    top_k: int = 5
) -> Dict:
    """
    Runs real-time inference on an input leaf image using the specified checkpoint.
    Returns:
      - Top-K class predictions with probabilities
      - Crop name & condition
      - Severity & Agronomic remedy
      - Inference latency (ms)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join("results/checkpoints", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        # Look for partial matches
        candidates = [f for f in os.listdir("results/checkpoints") if checkpoint_name in f]
        if candidates:
            checkpoint_path = os.path.join("results/checkpoints", candidates[0])

    model_name = infer_model_name_from_checkpoint(checkpoint_path)
    model = get_or_load_model(checkpoint_path, model_name=model_name, device=device)

    # Preprocessing
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_transforms(image_size=224, is_train=False)
    input_tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
    latency_ms = round((time.time() - start_time) * 1000.0, 2)

    _, idx_to_class = load_classes_mapping()

    # Get top-k indices
    top_indices = probabilities.argsort()[-top_k:][::-1]
    top_predictions = []

    for rank, idx in enumerate(top_indices):
        cls_name = idx_to_class.get(int(idx), f"Unknown_Class_{idx}")
        prob = float(probabilities[idx])

        # Parse crop and disease name
        parts = cls_name.split("___")
        crop_name = parts[0].replace("_", " ")
        disease_name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

        top_predictions.append({
            "rank": rank + 1,
            "class_raw": cls_name,
            "crop": crop_name,
            "disease": disease_name,
            "probability": round(prob, 4),
            "percentage": f"{prob * 100.0:.2f}%",
            "is_healthy": "healthy" in disease_name.lower()
        })

    top_result = top_predictions[0]
    top_class = top_result["class_raw"]

    advisory = REMEDIES.get(top_class, {
        "severity": "Unknown",
        "description": f"Classification result for {top_result['crop']} - {top_result['disease']}.",
        "remedy": "Isolate affected crops, consult local agricultural extension specialist, and monitor spread."
    })

    return {
        "model_architecture": model_name,
        "checkpoint_used": os.path.basename(checkpoint_path),
        "device": device.type,
        "latency_ms": latency_ms,
        "top_prediction": top_result,
        "all_predictions": top_predictions,
        "advisory": advisory
    }
