
def generate_submission(model, X_test, submission_template, output_path):
    submission_template['target'] = model.predict(X_test)
    submission_template.to_csv(output_path, index=False)
