import clip
import numpy as np
import os
import torch
from pathlib import Path
from tqdm.auto import tqdm


CLIP_MODEL_NAME = "ViT-L/14@336px"
CLIP_MODEL_FILE = "ViT-L-14-336px.pt"
_CLIP_MODEL_CACHE = {}


def _project_root():
    return Path(os.environ.get("ROOT_DIR", Path(__file__).resolve().parents[1])).resolve()


def _clip_model_path():
    return Path(os.environ.get("CLIP_MODEL_PATH", _project_root() / "clip" / CLIP_MODEL_FILE)).resolve()


def _load_clip_model(device):
    cache_key = str(device)
    if cache_key in _CLIP_MODEL_CACHE:
        return _CLIP_MODEL_CACHE[cache_key]

    model_path = _clip_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"CLIP checkpoint not found: {model_path}. "
            f"Upload {CLIP_MODEL_FILE} to {_project_root() / 'clip'} or set CLIP_MODEL_PATH."
        )

    print(f"Loading CLIP checkpoint from {model_path}", flush=True)
    model, _ = clip.load(str(model_path), device=device)
    model.eval()
    _CLIP_MODEL_CACHE[cache_key] = model
    return model


def get_clip_features(raw_text: str):
    '''
    return clip features for a given text prompt
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_clip_model(device)
    text = clip.tokenize([raw_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    # normalize the features
    text_features /= torch.norm(text_features, dim=-1, keepdim=True)
    assert text_features.shape == (1, 768)

    return text_features.detach().float()


def get_clip_features_loop(language_motion_dict: dict, range_start, range_end):

    language_motion_dict['clip_features'] = np.zeros((len(language_motion_dict['raw_text']), 768))

    for idx in tqdm(range(range_start, range_end)):
        text = language_motion_dict['raw_text'][idx]
        text_features = get_clip_features(raw_text=text).cpu().numpy()
        language_motion_dict['clip_features'][[idx]] = text_features
        print(idx, flush=True)

    print(f'language_motion_dict["clip_features"] shape: {language_motion_dict["clip_features"].shape}', flush=True)

    assert len(language_motion_dict.keys()) == 4, 'language_motion_dict should have 4 keys'

    return language_motion_dict
