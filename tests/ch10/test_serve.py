from fastapi.testclient import TestClient

from app.core.taxonomy import NUM_CLASSES, TOPIC_NAMES


def test_classify_validates_input_and_returns_all_scores():
    from scripts.ch10.serve import create_app

    class FakeRuntime:
        def classify(self, texts):
            return [{"labels": [TOPIC_NAMES[0]],
                     "scores": {name: (0.9 if i == 0 else 0.1)
                                for i, name in enumerate(TOPIC_NAMES)}} for _ in texts]

    with TestClient(create_app(runtime=FakeRuntime())) as client:
        assert client.post("/classify", json={"texts": []}).status_code == 422
        response = client.post("/classify", json={"texts": ["我的猫粮何时送到"]})
        assert response.status_code == 200
        assert len(response.json()["results"][0]["scores"]) == NUM_CLASSES
        assert response.json()["results"][0]["labels"] == [TOPIC_NAMES[0]]
