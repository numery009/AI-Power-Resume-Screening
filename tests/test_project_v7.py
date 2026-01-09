import pytest

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    pytest.skip("numpy and pandas are required for these tests", allow_module_level=True)
from conftest import import_project_v7


def test_f32(tmp_path):
    module = import_project_v7(tmp_path)
    data = module._f32([1, 2, 3])
    assert data.dtype == np.float32
    assert data.shape == (3,)


def test_is_good_token_filters(tmp_path):
    module = import_project_v7(tmp_path)
    module.name_list = {"john"}
    module.stop_words = {"the"}
    module.irrelevant_tokens = {"resume"}
    assert module._is_good_token("python") is True
    assert module._is_good_token("john") is False
    assert module._is_good_token("the") is False
    assert module._is_good_token("resume") is False
    assert module._is_good_token("a") is False


def test_load_dataset(tmp_path, monkeypatch):
    module = import_project_v7(tmp_path)
    data = pd.DataFrame({"Resume": ["a", "b"], "Category": ["X", "Y"]})
    dataset_path = tmp_path / "resume_dataset.csv"
    data.to_csv(dataset_path, index=False)
    original_read_csv = pd.read_csv
    monkeypatch.setattr(module.pd, "read_csv", lambda _path: original_read_csv(dataset_path))
    df, X, y_encoded, y_raw, encoder = module.load_dataset()
    assert list(X) == ["a", "b"]
    assert list(y_raw) == ["X", "Y"]
    assert set(encoder.classes_) == {"X", "Y"}
    assert df.shape[0] == 2


def test_extract_text_pdf(tmp_path):
    module = import_project_v7(tmp_path)
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("placeholder")
    text = module.extract_text(str(pdf_path))
    assert text == "PDF content"


def test_extract_text_docx(tmp_path):
    module = import_project_v7(tmp_path)
    docx_path = tmp_path / "sample.docx"
    docx_path.write_text("placeholder")
    text = module.extract_text(str(docx_path))
    assert text == "DOCX content"


def test_extract_resume_entities(tmp_path):
    module = import_project_v7(tmp_path)
    class DummyEnt:
        def __init__(self, text, label_):
            self.text = text
            self.label_ = label_
    class DummyDoc:
        def __init__(self, ents):
            self.ents = ents
    module.nlp = lambda _text: DummyDoc(ents=[DummyEnt("5 years", "DATE")])
    text = "Bachelor of Science\nExperience: 5 years"
    result = module.extract_resume_entities(text)
    assert result["Experience"] == "5 years"
    assert "Bachelor of Science" in result["Education"]


def test_get_cached_embeddings(tmp_path):
    module = import_project_v7(tmp_path)
    module.sbert_model.encode = lambda _text, convert_to_numpy=True: np.array([1, 2, 3], dtype=np.float64)
    vec = module.get_cached_embeddings("hello")
    assert vec.dtype == np.float32
    assert np.allclose(vec, np.array([1, 2, 3], dtype=np.float32))


def test_train_xgb_model(tmp_path, monkeypatch):
    module = import_project_v7(tmp_path)
    df = pd.DataFrame({
        "Resume": [f"resume {i}" for i in range(10)],
        "Category": ["X", "Y"] * 5,
    })
    y_encoded = np.array([0, 1] * 5)
    monkeypatch.setattr(
        module,
        "load_dataset",
        lambda: (df, df["Resume"], y_encoded, df["Category"].values, module.LabelEncoder().fit(df["Category"])),
    )
    monkeypatch.setattr(module, "get_cached_embeddings", lambda _text: np.array([0.1, 0.2], dtype=np.float32))
    model, encoder, X_train, X_test, y_test = module.train_xgb_model()
    assert X_train.shape[1] == 2
    assert len(encoder.classes_) == 2
    assert len(y_test) > 0


def test_evaluate_and_report(tmp_path):
    module = import_project_v7(tmp_path)
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)
    label_encoder = type("Enc", (), {"classes_": np.array(["A", "B"])})
    X_test = np.array([[0, 1], [1, 0]], dtype=np.float32)
    y_test = np.array([0, 1])
    report_df = module.evaluate_and_report(DummyModel(), X_test, y_test, label_encoder)
    assert "accuracy" in report_df.index


def test_interpret_keywords(tmp_path):
    module = import_project_v7(tmp_path)
    module._is_good_token = lambda token: True
    result = module.interpret_keywords("python data science", "data", top_k=2)
    assert len(result) <= 2
    assert all(isinstance(v, float) for v in result.values())


def test_get_openai_client(tmp_path, monkeypatch):
    module = import_project_v7(tmp_path)
    monkeypatch.setattr(module.os, "getenv", lambda key, default=None: "key" if key == "OPENAI_API_KEY" else default)
    module._OPENAI_AVAILABLE = True
    module.OpenAI = lambda api_key: f"client:{api_key}"
    client = module._get_openai_client()
    assert client == "client:key"


def test_llm_predict_label_openai(tmp_path):
    module = import_project_v7(tmp_path)
    class DummyClient:
        def __init__(self):
            self.chat = type("Chat", (), {
                "completions": type("Comp", (), {
                    "create": lambda *_args, **_kwargs: type("Resp", (), {
                        "choices": [type("Choice", (), {"message": type("Msg", (), {"content": '{"label": "B"}'})()})]
                    })()
                })()
            })()
    label = module.llm_predict_label_openai(DummyClient(), "text", ["A", "B"], model="gpt")
    assert label == "B"


def test_evaluate_llm_on_dataset_sample(tmp_path, monkeypatch):
    module = import_project_v7(tmp_path)
    df = pd.DataFrame({"Resume": ["a", "b"], "Category": ["A", "B"]})
    monkeypatch.setattr(module, "_get_openai_client", lambda: object())
    monkeypatch.setattr(module, "load_dataset", lambda: (df, df["Resume"], np.array([0, 1]), df["Category"].values, module.LabelEncoder().fit(df["Category"])))
    monkeypatch.setattr(module, "llm_predict_label_openai", lambda *_args, **_kwargs: "A")
    report_df, cm, labels = module.evaluate_llm_on_dataset_sample(2, "gpt")
    assert report_df.shape[0] > 0
    assert cm.shape[0] == len(labels)


def test_llm_match_score(tmp_path, monkeypatch):
    module = import_project_v7(tmp_path)
    module._OPENAI_AVAILABLE = True
    monkeypatch.setattr(module.os, "getenv", lambda key, default=None: "key" if key == "OPENAI_API_KEY" else default)
    class DummyClient:
        def __init__(self, *_, **__):
            self.chat = type("Chat", (), {
                "completions": type("Comp", (), {
                    "create": lambda *_args, **_kwargs: type("Resp", (), {
                        "choices": [type("Choice", (), {"message": type("Msg", (), {"content": '{"match_score": 0.8, "category": "Data Science"}'})()})]
                    })()
                })()
            })()
    module.OpenAI = DummyClient
    score, category, latency = module.llm_match_score("resume", "job")
    assert score == 0.8
    assert category == "Data Science"
    assert latency >= 0.0


def test_match_resume_with_job(tmp_path, monkeypatch):
    module = import_project_v7(tmp_path)
    monkeypatch.setattr(module, "get_cached_embeddings", lambda text: np.array([1.0, 0.0], dtype=np.float32) if text == "resume" else np.array([0.0, 1.0], dtype=np.float32))
    score = module.match_resume_with_job("resume", "job")
    assert score == 0.0


def test_compute_class_distribution(tmp_path, monkeypatch):
    module = import_project_v7(tmp_path)
    df = pd.DataFrame({"Resume": ["a", "b"], "Category": ["A", "B"]})
    monkeypatch.setattr(module, "load_dataset", lambda: (df, df["Resume"], np.array([0, 1]), df["Category"].values, module.LabelEncoder().fit(df["Category"])))
    _, summary, total_rows, n_classes, _ = module.compute_class_distribution()
    assert total_rows == 2
    assert n_classes == 2
    assert set(summary["Category"]) == {"A", "B"}


def test_render_class_distribution(tmp_path, monkeypatch):
    module = import_project_v7(tmp_path)
    summary = pd.DataFrame({"Category": ["A", "B"], "Count": [1, 2]})
    monkeypatch.chdir(tmp_path)
    module.render_class_distribution(summary)
    assert (tmp_path / "class_distribution.csv").exists()


def test_badges_row(tmp_path):
    module = import_project_v7(tmp_path)
    called = {}
    module.st.markdown = lambda text: called.setdefault("text", text)
    module.badges_row(deterministic=True, explainable=False, offline=True)
    assert "Reproducible" in called["text"]


def test_ethics_callout(tmp_path):
    module = import_project_v7(tmp_path)
    called = {}
    module.st.info = lambda text: called.setdefault("text", text)
    module.ethics_callout(run_llm_baseline=True)
    assert "token costs" in called["text"]


def test_resume_screening_dashboard_smoke(tmp_path):
    module = import_project_v7(tmp_path)
    module.resume_screening_dashboard()
