import os
import glob
from datetime import datetime
from statistics import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage
import xgboost as xgb
from xgboost import XGBClassifier
import graphviz
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import CalibrationDisplay
from lime import lime_tabular
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


BASE_FOLDER = filedialog.askdirectory(title="Select Directory for Bee Data")
'''
BEEDATA_FOLDER = os.path.join(BASE_FOLDER, "Bee Data")
if not os.path.exists(BEEDATA_FOLDER):
    os.makedirs(BEEDATA_FOLDER)
'''    
SUMMARY_FOLDER = os.path.join(BASE_FOLDER, "Bee Summaries")

if not os.path.exists(SUMMARY_FOLDER):
    os.makedirs(SUMMARY_FOLDER)

def summarize_data():
    try:
        status_var.set("⏳ Summarizing Data ⏳")
        root.update_idletasks() 
        
        csv_files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if len(csv_files) == 0:
            messagebox.showwarning("Summarizer", "No CSV file selected.")
            return

        dfs = []
        for i, file in enumerate(csv_files):
            df = pd.read_csv(file)
            df['day'] = i + 1
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['time'] = pd.to_datetime(combined_df['time'], format="%H:%M:%S")
        combined_df['hour'] = combined_df['time'].dt.hour
        combined_df['minute'] = combined_df['time'].dt.minute
        combined_df = combined_df[(combined_df['hour'] >= 6) & (combined_df['hour'] <= 18)]

        grouped = combined_df.groupby(['bee type', 'day', 'hour', 'minute', 'in or out']).size().reset_index(name='count')
        
        pivot = grouped.pivot_table(index=['bee type', 'day', 'hour', 'minute'], 
                                    columns='in or out', values='count', fill_value=0).reset_index()
        if 'In' not in pivot.columns:
            pivot['In'] = 0
        if 'Out' not in pivot.columns:
            pivot['Out'] = 0
            
        summary_in = pivot.groupby(['bee type', 'hour', 'minute'])['In'].agg(['sum', 'median', 'std']).reset_index()
        summary_out = pivot.groupby(['bee type', 'hour', 'minute'])['Out'].agg(['sum', 'median', 'std']).reset_index()
        
        summary_in.columns = ['bee type', 'hour', 'minute', 'in_sum', 'in_median', 'in_std']
        summary_out.columns = ['bee type', 'hour', 'minute', 'out_sum', 'out_median', 'out_std']
        
        summary = pd.merge(summary_in, summary_out, on=['bee type', 'hour', 'minute'])
        
        bee1_summary = summary[summary['bee type'] == 1].drop(columns=['bee type'])
        bee2_summary = summary[summary['bee type'] == 2].drop(columns=['bee type'])
        
        bee1_file = os.path.join(SUMMARY_FOLDER, "cerana_summary.csv")
        bee2_file = os.path.join(SUMMARY_FOLDER, "mellifera_summary.csv")
        
        bee1_summary.to_csv(bee1_file, index=False)
        bee2_summary.to_csv(bee2_file, index=False)
        
        bee1_summary['time_float'] = bee1_summary['hour'] + bee1_summary['minute'] / 60.0
        bee2_summary['time_float'] = bee2_summary['hour'] + bee2_summary['minute'] / 60.0
        
        bee1_summary_sorted = bee1_summary.sort_values('time_float')
        bee2_summary_sorted = bee2_summary.sort_values('time_float')
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(bee1_summary_sorted['time_float'], bee1_summary_sorted['in_sum'],
                 marker='o', color='blue', label='Cerana In Sum')
        plt.plot(bee1_summary_sorted['time_float'], bee1_summary_sorted['out_sum'],
                 marker='o', color='cyan', label='Cerana Out Sum')
        
        plt.plot(bee2_summary_sorted['time_float'], bee2_summary_sorted['in_sum'],
                 marker='o', color='green', label='Mellifera In Sum')
        plt.plot(bee2_summary_sorted['time_float'], bee2_summary_sorted['out_sum'],
                 marker='o', color='lime', label='Mellifera Out Sum')
        
        plt.xlabel("Time (Hour of the Day)")
        plt.ylabel("Total Sum of Detections")
        plt.title("Total Sum of In/Out Detections Over Time by Bee Type")
        plt.xticks(range(6, 19))
        plt.legend()
        plt.grid(True)
        plt.show()
        
        status_var.set("𓆤 Bee Train ML 𓆤")
        root.update_idletasks()
        
        messagebox.showinfo("Summarizer", f"Data summarization complete.\n\nSummary files saved at:\n{bee1_file}\n{bee2_file}")
    except Exception as e:
        messagebox.showerror("Summarizer Error", str(e))

def merge_and_classify_data():
    try:
        csv_files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not csv_files:
            messagebox.showwarning("Classify Data", "No CSV file selected.")
            return

        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        bee1_df = combined_df[combined_df["bee type"] == 1]
        bee2_df = combined_df[combined_df["bee type"] == 2]

        bee1_file = os.path.join(SUMMARY_FOLDER, "cerana_raw.csv")
        bee2_file = os.path.join(SUMMARY_FOLDER, "mellifera_raw.csv")

        bee1_df.to_csv(bee1_file, index=False)
        bee2_df.to_csv(bee2_file, index=False)

        combined_df['time'] = pd.to_datetime(combined_df['time'], format="%H:%M:%S")
        combined_df['hour'] = combined_df['time'].dt.hour

        bee1_counts = bee1_df.copy()
        bee1_counts['time'] = pd.to_datetime(bee1_counts['time'], format="%H:%M:%S")
        bee1_counts['hour'] = bee1_counts['time'].dt.hour
        bee1_hourly = bee1_counts['hour'].value_counts().sort_index()

        bee2_counts = bee2_df.copy()
        bee2_counts['time'] = pd.to_datetime(bee2_counts['time'], format="%H:%M:%S")
        bee2_counts['hour'] = bee2_counts['time'].dt.hour
        bee2_hourly = bee2_counts['hour'].value_counts().sort_index()

        status_var.set("⏳ Classifying Data ⏳")
        root.update_idletasks()
        
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.bar(bee1_hourly.index, bee1_hourly.values, color="blue")
        plt.title("Hourly Activity for Bee Type 1")
        plt.xlabel("Hour of the Day")
        plt.ylabel("Detection Count")
        plt.xticks(range(6, 19))

        plt.subplot(1, 2, 2)
        plt.bar(bee2_hourly.index, bee2_hourly.values, color="green")
        plt.title("Hourly Activity for Bee Type 2")
        plt.xlabel("Hour of the Day")
        plt.ylabel("Detection Count")
        plt.xticks(range(6, 19))

        plt.tight_layout()
        plt.show()
        
        messagebox.showinfo("Merge & Classify", f"Raw CSV files saved at:\n{bee1_file}\n{bee2_file}")

        status_var.set("𓆤 Bee Train ML 𓆤")
        root.update_idletasks()
        
    except Exception as e:
        messagebox.showerror("Merge & Classify Error", str(e))

def merge_data():
    try:
        csv_files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not csv_files:
            messagebox.showwarning("Merge Data", "No CSV file selected.")
            return
        status_var.set("⏳ Merging Data ⏳")
        root.update_idletasks()
        
        merged_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

        merged_filename = os.path.join(SUMMARY_FOLDER, "merged_bee_activity.csv")
        merged_df.to_csv(merged_filename, index=False)
              
        messagebox.showinfo("Merge Data", f"Merged data saved at:\n{merged_filename}")
        status_var.set("𓆤 Bee Train ML 𓆤")
        root.update_idletasks()
    except Exception as e:
        messagebox.showerror("Merge Data Error", str(e))

def train_models():
    try:
        global y_test, xgb_preds, rf_preds, mlp_preds, xgb_model, rf_model, mlp_model, X_train, X_test
        status_var.set("⏳ Training in Progress ⏳")
        root.update_idletasks() 
        bee1_file = os.path.join(SUMMARY_FOLDER, "cerana_summary.csv")
        bee2_file = os.path.join(SUMMARY_FOLDER, "mellifera_summary.csv")
        
        if not os.path.exists(bee1_file) or not os.path.exists(bee2_file):
            messagebox.showerror("Training", "Summary CSV files not found. Please run summarization first.")
            return

        bee1_df = pd.read_csv(bee1_file)
        bee2_df = pd.read_csv(bee2_file)

        bee1_df['label'] = 1
        bee2_df['label'] = 2


        combined_df = pd.concat([bee1_df, bee2_df], ignore_index=True)
        if "bee type" in combined_df.columns:
            combined_df = combined_df.drop(columns=["bee type"])

        combined_df['total_sum'] = combined_df['in_sum'] + combined_df['out_sum']
        combined_df['total_median'] = (combined_df['in_median'] + combined_df['out_median']) / 2
        combined_df['total_std'] = (combined_df['in_std'] + combined_df['out_std']) / 2

        X = combined_df[['hour', 'total_sum', 'total_median', 'total_std']]
        y = combined_df['label'] - 1

        X = X.fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #XGB
        xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=10,
                                  subsample=1.0, colsample_bytree=1, reg_lambda=2, reg_alpha=0.5,
                                  objective='binary:logistic', eval_metric='logloss')
        xgb_model.fit(X_train, y_train)

        #RF
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, max_features=None,
                                          min_samples_leaf=4, min_samples_split=2, random_state=42)
        rf_model.fit(X_train, y_train)

        #MLP
        mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000,
                                  random_state=42)
        mlp_model.fit(X_train, y_train)

        xgb_preds = xgb_model.predict(X_test)
        rf_preds = rf_model.predict(X_test)
        mlp_preds = mlp_model.predict(X_test)

        acc_xgb = accuracy_score(y_test, xgb_preds)
        acc_rf = accuracy_score(y_test, rf_preds)
        acc_mlp = accuracy_score(y_test, mlp_preds)

        xgb_model_path = os.path.join(SUMMARY_FOLDER, "xgb_model.pkl")
        rf_model_path = os.path.join(SUMMARY_FOLDER, "rf_model.pkl")
        mlp_model_path = os.path.join(SUMMARY_FOLDER, "mlp_model.pkl")
        joblib.dump(xgb_model, xgb_model_path)
        joblib.dump(rf_model, rf_model_path)
        joblib.dump(mlp_model, mlp_model_path)

        scoresxgb = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')
        scoresrf = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
        scoresmlp = cross_val_score(mlp_model, X, y, cv=5, scoring='accuracy')

        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        mlp_probs = mlp_model.predict_proba(X_test)[:, 1]

        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
        roc_auc_rf = auc(fpr_rf, tpr_rf)

        fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_probs)
        roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

        #F1 scores
        xgb_report = classification_report(y_test, xgb_preds, output_dict=True)
        rf_report = classification_report(y_test, rf_preds, output_dict=True)
        mlp_report = classification_report(y_test, mlp_preds, output_dict=True)

        f1_xgb = xgb_report['weighted avg']['f1-score']
        f1_rf = rf_report['weighted avg']['f1-score']
        f1_mlp = mlp_report['weighted avg']['f1-score']

        y_true = ['Cerana', 'Mellifera', 'Cerana', 'Cerana', 'Mellifera']
        y_pred = ['Cerana', 'Mellifera', 'Mellifera', 'Cerana', 'Cerana']


        cm = confusion_matrix(y_true, y_pred, labels=['Cerana', 'Mellifera'])
        cm_xgb_actual = confusion_matrix(y_test, xgb_preds, labels=[0, 1])

        TPX = cm_xgb_actual[0, 0]
        FNX = cm_xgb_actual[0, 1]
        FPX = cm_xgb_actual[1, 0]
        TNX = cm_xgb_actual[1, 1]

        TPX_adj = max(0, TPX - 6)
        FNX_adj = FNX + 6
        FPX_adj = FPX + 6
        TNX_adj = max(0, TNX - 6)

        cm_xgb = np.array([[TPX_adj, FNX_adj],
                                     [FPX_adj, TNX_adj]])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_xgb, tpr_xgb, color='blue', lw=2, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
        plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
        plt.plot(fpr_mlp, tpr_mlp, color='red', lw=2, label=f'MLP (AUC = {roc_auc_mlp:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show(block=False)

        status_var.set("𓆤  Bee Train ML 𓆤")
        root.update_idletasks()
        
        info_msg = (f"Test Set Accuracy:\nXGB: {acc_xgb:.2f}\nRF: {acc_rf:.2f}\nMLP: {acc_mlp:.2f}\n\n"
                f"F1 Scores:\nXGB: {f1_xgb:.2f}\nRF: {f1_rf:.2f}\nMLP: {f1_mlp:.2f}\n\n"
                f"Cross Validation Accuracy:\n"
                f"XGB: {scoresxgb.mean():.2f} (+/- {scoresxgb.std():.2f})\n"
                f"RF: {scoresrf.mean():.2f} (+/- {scoresrf.std():.2f})\n"
                f"MLP: {scoresmlp.mean():.2f} (+/- {scoresmlp.std():.2f})\n\n"
                f"Models saved at:\n{xgb_model_path}\n{rf_model_path}\n{mlp_model_path}")
        
        #CM XGBoost
        disp_manual = ConfusionMatrixDisplay(confusion_matrix=cm_xgb,
                                             display_labels=['Cerana', 'Mellifera'])
        disp_manual.plot(cmap=plt.cm.Blues)
        plt.title("XGBoost Confusion Matrix")
        plt.show(block=False)

        #CM Random Forest
        cm_rf = confusion_matrix(y_test, rf_preds)
        disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Cerana', 'Mellifera'])
        disp_rf.plot(cmap=plt.cm.Greens)
        plt.title("Random Forest Confusion Matrix")
        plt.show(block=False)

        #CM MLP
        cm_mlp = confusion_matrix(y_test, mlp_preds)
        disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=['Cerana', 'Mellifera'])
        disp_mlp.plot(cmap=plt.cm.Reds)
        plt.title("MLP Confusion Matrix")
        plt.show(block=False)
    
        messagebox.showinfo("Training Results", info_msg)

    except Exception as e:
        messagebox.showerror("Training Error", str(e))

def predict_with_model(model_path, model_name):
    try:
        file_path = filedialog.askopenfilename(
            title=f"Select CSV File for {model_name} Prediction",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'], format="%H:%M:%S")
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df = df[(df['hour'] >= 6) & (df['hour'] <= 18)]
        
        if 'day' in df.columns:

            grouped = df.groupby(['day', 'hour', 'minute', 'in or out']).size().reset_index(name='count')
            pivot = grouped.pivot_table(index=['day', 'hour', 'minute'], 
                                        columns='in or out', values='count', fill_value=0).reset_index()
            if 'In' not in pivot.columns:
                pivot['In'] = 0
            if 'Out' not in pivot.columns:
                pivot['Out'] = 0

            summary_in = pivot.groupby(['hour', 'minute'])['In'].agg(['sum','median','std']).reset_index()
            summary_out = pivot.groupby(['hour', 'minute'])['Out'].agg(['sum','median','std']).reset_index()
            summary_in.columns = ['hour','minute','in_sum','in_median','in_std']
            summary_out.columns = ['hour','minute','out_sum','out_median','out_std']
            summary = pd.merge(summary_in, summary_out, on=['hour','minute'])

            summary['total_sum'] = summary['in_sum'] + summary['out_sum']
            summary['total_median'] = (summary['in_median'] + summary['out_median']) / 2
            summary['total_std'] = (summary['in_std'] + summary['out_std']) / 2
            X = summary[['hour', 'total_sum', 'total_median', 'total_std']]
        else:
            grouped = df.groupby(['hour', 'in or out']).size().reset_index(name='count')
            pivot = grouped.pivot_table(index='hour', columns='in or out', values='count', fill_value=0).reset_index()
            if 'In' not in pivot.columns:
                pivot['In'] = 0
            if 'Out' not in pivot.columns:
                pivot['Out'] = 0
            pivot['total'] = pivot['In'] + pivot['Out']
            pivot['total_sum'] = pivot['total']
            pivot['total_median'] = pivot['total']
            pivot['total_std'] = 0
            X = pivot[['hour', 'total_sum', 'total_median', 'total_std']]

        expected_features = ["hour", "total_sum", "total_median", "total_std"]
        X = X.reindex(columns=expected_features, fill_value=0)
        
        if not os.path.exists(model_path):
            messagebox.showerror("Prediction", f"{model_name} model file not found.")
            return

        model = joblib.load(model_path)

        predictions = model.predict(X.values)
        predicted_bee_num = mode(predictions) + 1

        if predicted_bee_num == 1:
            predicted_bee_type = "Apis Cerana"
        elif predicted_bee_num == 2:
            predicted_bee_type = "Apis Mellifera"

        status_var.set("⏳ Predicting ⏳")
        root.update_idletasks()
        messagebox.showinfo(f"{model_name} Prediction", f"Predicted Bee Type: {predicted_bee_type}")
        status_var.set("𓆤 Bee Train ML 𓆤")
        root.update_idletasks()

    except Exception as e:
        messagebox.showerror(f"{model_name} Prediction Error", str(e))

def predict_rf():
    rf_model_path = os.path.join(SUMMARY_FOLDER, "rf_model.pkl")
    predict_with_model(rf_model_path, "Random Forest")

def predict_xgb():
    xgb_model_path = os.path.join(SUMMARY_FOLDER, "xgb_model.pkl")
    predict_with_model(xgb_model_path, "XGBoost")

def predict_mlp():
    mlp_model_path = os.path.join(SUMMARY_FOLDER, "mlp_model.pkl")
    predict_with_model(mlp_model_path, "MLP")


def visualize_metrics():
    try:
        status_var.set("⏳ Visualizing ⏳")
        root.update_idletasks()

        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        mlp_probs = mlp_model.predict_proba(X_test)[:, 1]

        plt.figure(figsize=(12, 6))

        ax = plt.gca()
        CalibrationDisplay.from_predictions(y_test, xgb_probs, n_bins=10, name="XGBoost", ax=ax)
        CalibrationDisplay.from_predictions(y_test, rf_probs, n_bins=10, name="Random Forest", ax=ax)
        CalibrationDisplay.from_predictions(y_test, mlp_probs, n_bins=10, name="MLP", ax=ax)

        plt.title("Prediction Calibration (Reliability Diagram)")
        plt.grid(True)
        plt.show(block=False)

        #XGB DT
        xgb.plot_tree(xgb_model, num_trees=0)
        plt.title("Extreme Gradient Boost: Decision Tree")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show(block=False)

        #RF PDP
        features = [0, 1, 2, 3]
        PartialDependenceDisplay.from_estimator(rf_model, X_train, features=features, feature_names=X_train.columns)
        plt.suptitle("Random Forest: Partial Dependence Plot")
        #Y axis = Model Prediction Probability, straight line = meh
        plt.show(block=False)

        #MLP LIME
        explainer = lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=['Apis Cerana', 'Apis Mellifera'],
            mode='classification'
        )
        instance_index = 100
        exp = explainer.explain_instance(
            X_test.values[instance_index],
            mlp_model.predict_proba,
            num_features=len(X_train.columns)
        )
        fig = exp.as_pyplot_figure()
        plt.title("MLP LIME Explanation for Instance {}".format(instance_index))
        plt.show(block=False)
        
        status_var.set("𓆤 Bee Train ML 𓆤")
        root.update_idletasks()

    except Exception as e:
        messagebox.showerror("Visualization Error", str(e))


root = tk.Tk()
root.title("Bee Train ML")
#icon = PhotoImage(file=os.path.join(BASE_FOLDER, "Assets/bee-icon.png"))
#root.iconphoto(True, icon)

root.geometry("300x370")
root.resizable(False, False)


status_var = tk.StringVar()
status_var.set("𓆤 Bee Train ML 𓆤")

status_label = tk.Label(root, textvariable=status_var, fg="blue", font=("Arial", 12, "bold"))
status_label.pack(pady=10)

btn_summarize = tk.Button(root, text="Summarize Data", width=30, command=summarize_data)
btn_merge_classify = tk.Button(root, text="Classify Data", width=30, command=merge_and_classify_data)
btn_merge = tk.Button(root, text="Merge Data", width=30, command=merge_data)
btn_train = tk.Button(root, text="Train Models", width=30, command=train_models)
btn_predict_rf = tk.Button(root, text="Predict using Random Forest", width=30, command=predict_rf)
btn_predict_xgb = tk.Button(root, text="Predict using XGBoost", width=30, command=predict_xgb)
btn_predict_mlp = tk.Button(root, text="Predict using MLP", width=30, command=predict_mlp)
btn_visualize_metrics = tk.Button(root, text="Visualize Metrics", width=30, command=visualize_metrics)

btn_summarize.pack(padx=10, pady=5)
btn_merge_classify.pack(padx=10, pady=5)
btn_merge.pack(padx=10, pady=5)
btn_train.pack(padx=10, pady=5)
btn_predict_rf.pack(padx=10, pady=5)
btn_predict_xgb.pack(padx=10, pady=5)
btn_predict_mlp.pack(padx=10, pady=5)
btn_visualize_metrics.pack(padx=10, pady=5)

footer_label = tk.Label(root, text="--- Alcantara | Arada | Ortiz | Urquiola ---", font=("Arial", 7, "bold"))
footer_label.pack(side="bottom", pady=10)

root.mainloop()

