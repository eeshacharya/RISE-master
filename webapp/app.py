"""
Flask server for XAI Saliency Comparison Lab.
Supports ~40 concurrent users with a job queue.
"""
import os, sys, uuid, threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISE_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, RISE_DIR)

from flask import Flask, request, jsonify, render_template
from methods import process_image, get_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED = {'jpg', 'jpeg', 'png', 'bmp', 'webp', 'gif'}

# ─── Job store ────────────────────────────────────────────────────────────────
jobs: dict = {}
jobs_lock = threading.Lock()

# Limit concurrent ML jobs so CPU isn't overwhelmed
_sem = threading.Semaphore(3)

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    f = request.files['image']
    ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
    if ext not in ALLOWED:
        return jsonify({'error': f'File type .{ext} not allowed'}), 400

    jid = str(uuid.uuid4())[:8]
    path = os.path.join(UPLOAD_DIR, f'{jid}.{ext}')
    f.save(path)

    with jobs_lock:
        jobs[jid] = {'status': 'queued', 'progress': 0}

    t = threading.Thread(target=_run_job, args=(jid, path), daemon=True)
    t.start()
    return jsonify({'job_id': jid})


@app.route('/status/<jid>')
def status(jid):
    with jobs_lock:
        j = jobs.get(jid)
    if not j:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status':   j['status'],
        'progress': j.get('progress', 0),
        'error':    j.get('error', ''),
    })


@app.route('/result/<jid>')
def result(jid):
    with jobs_lock:
        j = jobs.get(jid)
    if not j:
        return jsonify({'error': 'Job not found'}), 404
    if j['status'] != 'done':
        return jsonify({'error': 'Not ready yet'}), 202
    return jsonify(j['result'])


@app.route('/gallery')
def gallery():
    """Return lightweight metadata for all jobs (newest first)."""
    with jobs_lock:
        items = [
            {
                'job_id':     k,
                'status':     v['status'],
                'class_name': v.get('class_name', ''),
                'confidence': v.get('confidence', 0),
                'thumbnail':  v.get('thumbnail', ''),
            }
            for k, v in jobs.items()
        ]
    return jsonify(items[::-1])


# ─── Background worker ────────────────────────────────────────────────────────

def _run_job(jid, path):
    def cb(p):
        with jobs_lock:
            if jid in jobs:
                jobs[jid]['progress'] = int(p)

    _sem.acquire()
    try:
        with jobs_lock:
            jobs[jid]['status'] = 'processing'

        result = process_image(path, progress_cb=cb)

        with jobs_lock:
            jobs[jid] = {
                'status':     'done',
                'progress':   100,
                'class_name': result['class_name'],
                'confidence': result['confidence'],
                'thumbnail':  result['original_img'],
                'result':     result,
            }
    except Exception as e:
        import traceback; traceback.print_exc()
        with jobs_lock:
            jobs[jid] = {'status': 'error', 'progress': 0, 'error': str(e)}
    finally:
        _sem.release()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Pre-loading model weights (first run downloads ~100 MB)...")
    get_model()
    print("\n" + "="*55)
    print("  XAI Saliency Lab is running!")
    print("  Local URL:  http://localhost:8080")
    print("  Network:    http://0.0.0.0:8080")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
