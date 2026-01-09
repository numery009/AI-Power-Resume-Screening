import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.skip("numpy is required for these tests", allow_module_level=True)

os.environ.setdefault("MPLBACKEND", "Agg")


class DummyCache:
    def __call__(self, func=None, **_kwargs):
        if func is not None:
            return func
        def decorator(inner):
            return inner
        return decorator

    def clear(self):
        return None


class DummySidebar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def header(self, *_args, **_kwargs):
        return None

    def button(self, *_args, **_kwargs):
        return False

    def selectbox(self, _label, options, **_kwargs):
        return options[0]


class DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None
        return _noop

    def metric(self, *_args, **_kwargs):
        return None


class DummyProgress:
    def progress(self, *_args, **_kwargs):
        return None


class DummyStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.cache_data = DummyCache()
        self.cache_resource = DummyCache()
        self.sidebar = DummySidebar()

    def title(self, *_args, **_kwargs):
        return None

    def caption(self, *_args, **_kwargs):
        return None

    def subheader(self, *_args, **_kwargs):
        return None

    def header(self, *_args, **_kwargs):
        return None

    def dataframe(self, *_args, **_kwargs):
        return None

    def pyplot(self, *_args, **_kwargs):
        return None

    def image(self, *_args, **_kwargs):
        return None

    def write(self, *_args, **_kwargs):
        return None

    def markdown(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def success(self, *_args, **_kwargs):
        return None

    def progress(self, *_args, **_kwargs):
        return DummyProgress()

    def columns(self, n, *_args, **_kwargs):
        return [DummyColumn() for _ in range(n)]

    def text_area(self, *_args, **_kwargs):
        return ""

    def file_uploader(self, *_args, **_kwargs):
        return None

    def selectbox(self, _label, options, **_kwargs):
        return options[0]

    def button(self, *_args, **_kwargs):
        return False

    def number_input(self, _label, **kwargs):
        return kwargs.get("value", 0)


class DummySpacy(types.ModuleType):
    def __init__(self):
        super().__init__("spacy")

    def load(self, _name):
        return DummyNLP()

    def blank(self, _name):
        return DummyNLP()


class DummyEnt:
    def __init__(self, text, label_):
        self.text = text
        self.label_ = label_


class DummyDoc:
    def __init__(self, ents=None):
        self.ents = ents or []


class DummyNLP:
    def __call__(self, _text):
        return DummyDoc()


class DummyNLTK(types.ModuleType):
    def __init__(self):
        super().__init__("nltk")

    def download(self, *_args, **_kwargs):
        return True


class DummyCorpus(types.ModuleType):
    def __init__(self):
        super().__init__("nltk.corpus")
        self.names = types.SimpleNamespace(words=lambda: ["john", "mary"])
        self.stopwords = types.SimpleNamespace(words=lambda _lang="english": ["the", "and"])


class DummySentenceTransformers(types.ModuleType):
    class SentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, _text, convert_to_numpy=True):
            data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            return data


class DummyShap(types.ModuleType):
    class TreeExplainer:
        def __init__(self, *_args, **_kwargs):
            pass

        def shap_values(self, data):
            return np.zeros_like(data)


class DummyXGBoost(types.ModuleType):
    class XGBClassifier:
        def __init__(self, *_args, **_kwargs):
            self._classes = None

        def fit(self, _X, y):
            self._classes = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            n_samples = len(X)
            n_classes = len(self._classes) if self._classes is not None else 2
            return np.full((n_samples, n_classes), 1 / n_classes)

        def get_booster(self):
            return self


class DummyPdfPlumber(types.ModuleType):
    class DummyPage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class DummyPDF:
        def __init__(self, pages):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def open(self, _path):
        pages = [self.DummyPage("PDF content")]
        return self.DummyPDF(pages)


class DummyDocx(types.ModuleType):
    class DummyParagraph:
        def __init__(self, text):
            self.text = text

    class DummyDoc:
        def __init__(self):
            self.paragraphs = [DummyDocx.DummyParagraph("DOCX content")]

    def Document(self, _path):
        return self.DummyDoc()


class DummyOpenAI(types.ModuleType):
    class OpenAI:
        def __init__(self, *_, **__):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **_kwargs: types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"label": "A"}'))]
                    )
                )
            )


def import_project_v7(tmp_path):
    st_module = DummyStreamlit()
    spacy_module = DummySpacy()
    nltk_module = DummyNLTK()
    nltk_corpus_module = DummyCorpus()
    sentence_module = DummySentenceTransformers("sentence_transformers")
    shap_module = DummyShap("shap")
    xgboost_module = DummyXGBoost("xgboost")
    pdfplumber_module = DummyPdfPlumber("pdfplumber")
    docx_module = DummyDocx("docx")
    openai_module = DummyOpenAI("openai")

    sys.modules.update({
        "streamlit": st_module,
        "spacy": spacy_module,
        "nltk": nltk_module,
        "nltk.corpus": nltk_corpus_module,
        "sentence_transformers": sentence_module,
        "shap": shap_module,
        "xgboost": xgboost_module,
        "pdfplumber": pdfplumber_module,
        "docx": docx_module,
        "openai": openai_module,
    })

    module_path = Path(__file__).resolve().parents[1] / "project_v7.py"
    spec = importlib.util.spec_from_file_location("project_v7", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._test_tmp_path = tmp_path
    return module
