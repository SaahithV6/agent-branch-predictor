import json, pickle, sys

model, mlb = pickle.load(open(sys.argv[1], "rb"))
for line in sys.stdin:
    text = json.loads(line)["text"]
    probs = model.predict_proba([text])[0]
    print(json.dumps({"probs": dict(zip(mlb.classes_.tolist(), probs.round(4).tolist()))}), flush=True)
