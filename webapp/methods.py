"""
ML pipeline: RISE, LIME, GradCAM, and insertion/deletion AUC curves.
All computation runs on CPU (no CUDA required).
"""
import sys, os, threading

RISE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RISE_DIR)
os.chdir(RISE_DIR)

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import base64, io

DEVICE = torch.device('cpu')

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Model ────────────────────────────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("Loading ResNet-50 (ImageNet pretrained)...")
                try:
                    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                except Exception:
                    resnet = models.resnet50(pretrained=True)
                model = nn.Sequential(resnet, nn.Softmax(dim=1))
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                _model = model
                print("Model ready.")
    return _model

_labels = None
def get_class_name(c):
    global _labels
    if _labels is None:
        path = os.path.join(RISE_DIR, 'synset_words.txt')
        try:
            _labels = np.loadtxt(path, str, delimiter='\t')
        except Exception:
            return f'class_{c}'
    try:
        return ' '.join(_labels[c].split(',')[0].split()[1:])
    except Exception:
        return f'class_{c}'

def load_image(path):
    img = Image.open(path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    display = np.array(img.resize((224, 224))) / 255.0
    return img, tensor, display

# ─── Image rendering ──────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=96,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def image_to_b64(display):
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor('#07070f')
    ax.imshow(display)
    ax.axis('off')
    return _fig_to_b64(fig)

def saliency_to_b64(display, sal):
    s = sal.copy().astype(float)
    mn, mx = s.min(), s.max()
    if mx > mn:
        s = (s - mn) / (mx - mn)
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor('#07070f')
    ax.imshow(display)
    ax.imshow(s, cmap='jet', alpha=0.55, vmin=0, vmax=1)
    ax.axis('off')
    return _fig_to_b64(fig)

# ─── GradCAM ──────────────────────────────────────────────────────────────────

def gradcam_explain(model, img_tensor, target_class=None):
    """Thread-safe GradCAM on ResNet-50 layer4."""
    resnet = model[0]
    state = {}

    h1 = resnet.layer4.register_forward_hook(
        lambda m, i, o: state.update({'fmaps': o})
    )
    h2 = resnet.layer4.register_full_backward_hook(
        lambda m, gi, go: state.update({'grads': go[0]})
    )
    try:
        # Input must require grad to build computation graph (params are frozen)
        inp = img_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out = model(inp)
            if target_class is None:
                target_class = out.argmax(1).item()
            model.zero_grad()
            out[0, target_class].backward()

        fmaps = state['fmaps'].detach()
        grads = state['grads'].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * fmaps).sum(1)).squeeze().numpy()
        cam = resize(cam, (224, 224), anti_aliasing=True).astype(float)
        if cam.max() > 0:
            cam /= cam.max()
        return cam, target_class
    finally:
        h1.remove()
        h2.remove()

# ─── RISE ─────────────────────────────────────────────────────────────────────

def rise_explain(model, img_tensor, target_class=None,
                 n_masks=300, s=8, p1=0.5, batch=100, progress_cb=None):
    H, W = 224, 224
    cell = np.ceil(np.array([H, W]) / s).astype(int)
    up = (s + 1) * cell

    grid = (np.random.rand(n_masks, s, s) < p1).astype('float32')
    masks = np.empty((n_masks, H, W), dtype='float32')
    for i in range(n_masks):
        x = np.random.randint(0, cell[0])
        y = np.random.randint(0, cell[1])
        masks[i] = resize(grid[i], up, order=1, mode='reflect',
                          anti_aliasing=False)[x:x+H, y:y+W]
    masks_t = torch.from_numpy(masks.reshape(-1, 1, H, W)).float()

    if target_class is None:
        with torch.no_grad():
            target_class = model(img_tensor).argmax(1).item()

    sal = torch.zeros(H, W)
    for i in range(0, n_masks, batch):
        bm = masks_t[i:i+batch]
        with torch.no_grad():
            preds = model(bm * img_tensor)
        sal += (preds[:, target_class].view(-1, 1, 1) * bm.squeeze(1)).sum(0)
        if progress_cb:
            progress_cb(min(1.0, (i + batch) / n_masks))

    sal = sal.numpy() / n_masks / p1
    return sal, target_class

# ─── LIME ─────────────────────────────────────────────────────────────────────

def lime_explain(model, img_pil, img_tensor, target_class=None, num_samples=100):
    try:
        import warnings; warnings.filterwarnings('ignore')
        from lime import lime_image

        img_224 = np.array(img_pil.resize((224, 224)))

        def predict_fn(images):
            results = []
            for img in images:
                t = preprocess(Image.fromarray(img.astype(np.uint8))).unsqueeze(0)
                with torch.no_grad():
                    results.append(model(t).numpy()[0])
            return np.array(results)

        if target_class is None:
            with torch.no_grad():
                target_class = model(img_tensor).argmax(1).item()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_224, predict_fn, top_labels=5, hide_color=0, num_samples=num_samples
        )

        tc = target_class if target_class in explanation.local_exp else explanation.top_labels[0]
        d = dict(explanation.local_exp[tc])
        heatmap = np.vectorize(d.get)(explanation.segments).astype(float)
        mn, mx = heatmap.min(), heatmap.max()
        if mx > mn:
            heatmap = (heatmap - mn) / (mx - mn)
        return heatmap, target_class

    except Exception as e:
        print(f"LIME failed ({e}), using gradient×input fallback")
        return _gradient_saliency(model, img_tensor, target_class)

def _gradient_saliency(model, img_tensor, target_class=None):
    t = img_tensor.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        out = model(t)
        if target_class is None:
            target_class = out.argmax(1).item()
        out[0, target_class].backward()
    sal = (t.grad * t.detach()).abs().squeeze().numpy()
    sal = sal.max(axis=0)
    sal = resize(sal, (224, 224), anti_aliasing=True).astype(float)
    if sal.max() > 0:
        sal /= sal.max()
    return sal, target_class

# ─── AUC Curves ───────────────────────────────────────────────────────────────

_blur_kern = None
_blur_lock = threading.Lock()

def _get_blur_kern():
    global _blur_kern
    if _blur_kern is None:
        with _blur_lock:
            if _blur_kern is None:
                inp = np.zeros((11, 11))
                inp[5, 5] = 1
                k = gaussian_filter(inp, 5)
                kern = np.zeros((3, 3, 11, 11))
                kern[0, 0] = kern[1, 1] = kern[2, 2] = k
                _blur_kern = torch.from_numpy(kern.astype('float32'))
    return _blur_kern

def _blur(x):
    return nn.functional.conv2d(x, _get_blur_kern(), padding=5)

def auc_curve(model, img_tensor, saliency, mode='del', step=224):
    """Compute insertion/deletion curve matching the paper exactly.

    step=224 pixels per iteration → 224 steps → 225 score points,
    identical to CausalMetric(model, mode, 224, ...) in evaluation.py.
    """
    HW = 224 * 224
    n_steps = (HW + step - 1) // step  # = 224

    with torch.no_grad():
        target_class = model(img_tensor).argmax(1).item()

    # Work entirely in numpy; keep a contiguous C-order copy to avoid view issues
    img_np   = np.ascontiguousarray(img_tensor.numpy().reshape(1, 3, HW))
    zero_np  = np.zeros_like(img_np)           # deletion substrate (paper: zeros in norm. space)
    with torch.no_grad():
        blur_np = np.ascontiguousarray(
            _blur(img_tensor).numpy().reshape(1, 3, HW)
        )                                       # insertion substrate (blurred image)

    if mode == 'del':
        start_np  = img_np.copy()
        finish_np = zero_np
    else:
        start_np  = blur_np.copy()
        finish_np = img_np.copy()               # copy so finish_np is never mutated

    # Pixels ordered from most-salient to least-salient
    order = np.flip(np.argsort(saliency.reshape(-1))).copy()

    scores = []
    for i in range(n_steps + 1):
        t = torch.from_numpy(start_np.reshape(1, 3, 224, 224))
        with torch.no_grad():
            scores.append(float(model(t)[0, target_class].item()))
        if i < n_steps:
            coords = order[step * i : step * (i + 1)]
            start_np[0, :, coords] = finish_np[0, :, coords]

    scores = np.array(scores)
    # Trapezoid-rule AUC, normalised to [0,1] x-axis — matches evaluation.py auc()
    auc_val = float((scores.sum() - scores[0] / 2 - scores[-1] / 2) / n_steps)
    return scores.tolist(), round(auc_val, 4)

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def process_image(img_path, progress_cb=None):
    model = get_model()
    img_pil, img_tensor, display = load_image(img_path)

    with torch.no_grad():
        pred = model(img_tensor)
    target_class = int(pred.argmax(1).item())
    confidence = round(float(pred[0, target_class].item()) * 100, 2)
    class_name = get_class_name(target_class)
    if progress_cb: progress_cb(5)

    # GradCAM — fast (~1s)
    gc_sal, _ = gradcam_explain(model, img_tensor, target_class)
    if progress_cb: progress_cb(15)

    # LIME — medium (~5-10s)
    lime_sal, _ = lime_explain(model, img_pil, img_tensor, target_class, num_samples=100)
    if progress_cb: progress_cb(40)

    # RISE — ~10-15s
    def _rise_cb(p):
        if progress_cb:
            progress_cb(40 + int(p * 35))
    rise_sal, _ = rise_explain(model, img_tensor, target_class, n_masks=300, progress_cb=_rise_cb)
    if progress_cb: progress_cb(75)

    # AUC curves (3 methods × 2 modes = 6 calls)
    rise_del, rise_del_auc = auc_curve(model, img_tensor, rise_sal, 'del')
    rise_ins, rise_ins_auc = auc_curve(model, img_tensor, rise_sal, 'ins')
    if progress_cb: progress_cb(82)

    lime_del, lime_del_auc = auc_curve(model, img_tensor, lime_sal, 'del')
    lime_ins, lime_ins_auc = auc_curve(model, img_tensor, lime_sal, 'ins')
    if progress_cb: progress_cb(89)

    gc_del, gc_del_auc = auc_curve(model, img_tensor, gc_sal, 'del')
    gc_ins, gc_ins_auc = auc_curve(model, img_tensor, gc_sal, 'ins')
    if progress_cb: progress_cb(95)

    # Render images
    orig_b64  = image_to_b64(display)
    rise_b64  = saliency_to_b64(display, rise_sal)
    lime_b64  = saliency_to_b64(display, lime_sal)
    gc_b64    = saliency_to_b64(display, gc_sal)
    if progress_cb: progress_cb(100)

    return {
        'class_name':        class_name,
        'confidence':        confidence,
        'original_img':      orig_b64,
        'rise_img':          rise_b64,
        'lime_img':          lime_b64,
        'gradcam_img':       gc_b64,
        'rise_del_scores':   rise_del,
        'rise_ins_scores':   rise_ins,
        'rise_del_auc':      rise_del_auc,
        'rise_ins_auc':      rise_ins_auc,
        'lime_del_scores':   lime_del,
        'lime_ins_scores':   lime_ins,
        'lime_del_auc':      lime_del_auc,
        'lime_ins_auc':      lime_ins_auc,
        'gradcam_del_scores': gc_del,
        'gradcam_ins_scores': gc_ins,
        'gradcam_del_auc':   gc_del_auc,
        'gradcam_ins_auc':   gc_ins_auc,
    }
