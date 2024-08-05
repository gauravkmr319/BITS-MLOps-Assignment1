import joblib



model = joblib.load('model.pkl')

def test_model():
    print("pass")

if __name__ == "__main__":
    test_model()
