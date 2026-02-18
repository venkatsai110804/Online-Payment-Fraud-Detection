import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

try:
	from xgboost import XGBClassifier
except Exception:
	XGBClassifier = None

try:
	from imblearn.over_sampling import SMOTE
except Exception:
	SMOTE = None


def infer_feature_types(df, ignore_cols):
	numeric = df.select_dtypes(include=["number"]).columns.tolist()
	categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
	numeric = [c for c in numeric if c not in ignore_cols]
	categorical = [c for c in categorical if c not in ignore_cols]
	return numeric, categorical


def build_preprocessor(numeric_features, categorical_features):
	numeric_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler())
	])

	categorical_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		# `sparse` was deprecated; use `sparse_output` for newer scikit-learn
		("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features)
		],
		remainder="drop"
	)
	return preprocessor


def get_models(selected=None):
	all_models = {
		'random_forest': RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=42),
		'decision_tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
		'extratrees': ExtraTreesClassifier(n_jobs=-1, random_state=42),
		'svm': SVC(probability=True, class_weight='balanced', random_state=42),
	}
	if XGBClassifier is not None:
		all_models['xgboost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=42)

	if selected is None:
		return all_models

	sel = {}
	for name in selected:
		if name in all_models:
			sel[name] = all_models[name]
	return sel


def evaluate_model(model, X_test, y_test):
	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred, zero_division=0)
	try:
		proba = model.predict_proba(X_test)[:, 1]
		auc = roc_auc_score(y_test, proba)
	except Exception:
		auc = None
	return {'accuracy': acc, 'f1': f1, 'roc_auc': auc, 'report': classification_report(y_test, y_pred, zero_division=0)}


def RandomForest(X_train, X_test, y_train, y_test, **rfc_kwargs):
	"""Train a RandomForestClassifier and return metrics and model.

	Parameters:
	- X_train, X_test: feature DataFrames or arrays
	- y_train, y_test: target Series or arrays
	- rfc_kwargs: passed to RandomForestClassifier (e.g., n_estimators=100)

	Returns a dict with keys: model, train_accuracy, test_accuracy,
	confusion_matrix (pandas crosstab), report (classification_report), y_test_pred
	"""
	clf = RandomForestClassifier(**rfc_kwargs)
	clf.fit(X_train, y_train)

	y_test_pred = clf.predict(X_test)
	test_accuracy = accuracy_score(y_test, y_test_pred)

	y_train_pred = clf.predict(X_train)
	train_accuracy = accuracy_score(y_train, y_train_pred)

	cm = pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_test_pred, name='Predicted'))
	report = classification_report(y_test, y_test_pred, zero_division=0)

	return {
		'model': clf,
		'train_accuracy': train_accuracy,
		'test_accuracy': test_accuracy,
		'confusion_matrix': cm,
		'report': report,
		'y_test_pred': y_test_pred,
	}


def Decisiontree(X_train, X_test, y_train, y_test, **dt_kwargs):
	"""Train a DecisionTreeClassifier and return metrics and model.

	Parameters:
	- X_train, X_test: feature DataFrames or arrays
	- y_train, y_test: target Series or arrays
	- dt_kwargs: passed to DecisionTreeClassifier

	Returns a dict with keys: model, train_accuracy, test_accuracy,
	confusion_matrix (pandas crosstab), report (classification_report), y_test_pred
	"""
	clf = DecisionTreeClassifier(**dt_kwargs)
	clf.fit(X_train, y_train)

	y_test_pred = clf.predict(X_test)
	test_accuracy = accuracy_score(y_test, y_test_pred)

	y_train_pred = clf.predict(X_train)
	train_accuracy = accuracy_score(y_train, y_train_pred)

	cm = pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_test_pred, name='Predicted'))
	report = classification_report(y_test, y_test_pred, zero_division=0)

	return {
		'model': clf,
		'train_accuracy': train_accuracy,
		'test_accuracy': test_accuracy,
		'confusion_matrix': cm,
		'report': report,
		'y_test_pred': y_test_pred,
	}


def main(args):
	csv_path = args.csv
	target = args.target
	sample_size = args.sample
	save_path = args.save
	models_arg = args.models
	metric = args.metric
	use_smote = args.use_smote

	if not os.path.exists(csv_path):
		print(f"CSV file not found: {csv_path}")
		sys.exit(1)

	print(f"Reading up to {sample_size} rows from {csv_path}...")
	try:
		df = pd.read_csv(csv_path, nrows=sample_size)
	except Exception as e:
		print("Failed to read CSV sample:", e)
		sys.exit(1)

	if target not in df.columns:
		print(f"Target column '{target}' not found in sampled columns: {df.columns.tolist()}")
		sys.exit(1)

	print("Sample shape:", df.shape)

	# Drop high-cardinality features to avoid memory issues
	high_cardinality_cols = ['nameOrig', 'nameDest']
	df = df.drop(columns=[col for col in high_cardinality_cols if col in df.columns], errors='ignore')
	print(f"Dropped high-cardinality features: {high_cardinality_cols}")

	ignore_cols = [target]
	numeric_features, categorical_features = infer_feature_types(df, ignore_cols)

	print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
	print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

	X = df.drop(columns=[target])
	y = df[target].astype(int)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

	preprocessor = build_preprocessor(numeric_features, categorical_features)

	selected = None
	if models_arg:
		selected = [m.strip() for m in models_arg.split(',') if m.strip()]

	models = get_models(selected)
	if not models:
		print("No valid models selected. Exiting.")
		sys.exit(1)

	results = {}
	best_score = -1
	best_name = None
	best_pipeline = None

	for name, clf in models.items():
		print('\n' + '='*40)
		print(f"Training model: {name}")

		pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])

		X_tr = X_train.copy()
		y_tr = y_train.copy()

		# SMOTE branch: transform then resample then fit classifier
		if use_smote and SMOTE is not None:
			print('Applying SMOTE to training data...')
			preprocessor.fit(X_tr)
			X_tr_trans = preprocessor.transform(X_tr)
			sm = SMOTE(random_state=42)
			try:
				X_res, y_res = sm.fit_resample(X_tr_trans, y_tr)
				clf.fit(X_res, y_res)
				# build a pipeline that applies preprocessor then classifier (clf already fitted but sklearn will refit on fit call below if invoked)
				pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])
			except Exception as e:
				print(f"SMOTE failed: {e}. Training without SMOTE.")
				pipeline.fit(X_tr, y_tr)
		else:
			if use_smote and SMOTE is None:
				print('SMOTE requested but imbalanced-learn not installed. Skipping SMOTE.')
			pipeline.fit(X_tr, y_tr)

		# Evaluate
		try:
			metrics = evaluate_model(pipeline, X_test, y_test)
		except Exception as e:
			print(f"Evaluation failed for {name}: {e}")
			continue

		results[name] = metrics
		print(f"Results for {name}:")
		print(metrics['report'])
		print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']}")

		score_val = metrics.get(metric)
		if score_val is None:
			score_val = metrics.get('f1', 0)

		if score_val is not None and score_val > best_score:
			best_score = score_val
			best_name = name
			best_pipeline = pipeline

	print('\n' + '='*40)
	print('Summary of model performance:')
	for name, m in results.items():
		print(f"- {name}: accuracy={m['accuracy']:.4f}, f1={m['f1']:.4f}, roc_auc={m['roc_auc']}")

	if best_pipeline is not None:
		out_path = save_path
		with open(out_path, 'wb') as f:
			pickle.dump(best_pipeline, f)
		print(f"Best model '{best_name}' saved to {out_path} (metric={metric} -> {best_score})")
	else:
		print('No model was saved.')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train and compare multiple classifiers for fraud-detection")
	parser.add_argument("--csv", required=True, help="Path to transactions CSV")
	parser.add_argument("--target", required=True, help="Name of the target column (0/1)")
	parser.add_argument("--sample", type=int, default=100000, help="Number of rows to sample for training")
	parser.add_argument("--save", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "payments.pkl"), help="Path to save trained pipeline pickle")
	parser.add_argument("--models", help="Comma-separated list of models to train: random_forest,decision_tree,extratrees,svm,xgboost (default: all)")
	parser.add_argument("--metric", choices=['roc_auc','f1','accuracy'], default='roc_auc', help="Metric to choose best model")
	parser.add_argument("--use_smote", action='store_true', help="Apply SMOTE to training data (requires imbalanced-learn)")
	args = parser.parse_args()
	main(args)

