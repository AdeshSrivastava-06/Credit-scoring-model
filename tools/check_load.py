import joblib,traceback
files = ['preprocessor.pkl','label_encoder.pkl']
for f in files:
    print('\n--- Loading', f)
    try:
        obj = joblib.load(f)
        print('Loaded', f, 'type=', type(obj))
    except Exception:
        traceback.print_exc()
