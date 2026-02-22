import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum([p * np.log2(p) for p in probs if p > 0])


def info_gain(data, split_attribute_name, target_name="target"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = 0
    for v, c in zip(vals, counts):
        subset = data[data[split_attribute_name] == v]
        weighted_entropy += (c / counts.sum()) * entropy(subset[target_name])
    return total_entropy - weighted_entropy


def id3(data, original_data, features, target_name="target"):
    unique_targets = np.unique(data[target_name])
    if len(unique_targets) == 1:
        return unique_targets[0]

    if data.shape[0] == 0:
        return np.unique(original_data[target_name])[np.argmax(np.unique(original_data[target_name], return_counts=True)[1])]

    if len(features) == 0:
        return np.unique(data[target_name])[np.argmax(np.unique(data[target_name], return_counts=True)[1])]

    gains = [info_gain(data, feature, target_name) for feature in features]
    best_feature_idx = np.argmax(gains)
    best_feature = features[best_feature_idx]

    tree = {best_feature: {}}
    for val in np.unique(data[best_feature]):
        sub_data = data[data[best_feature] == val]
        sub_features = [f for f in features if f != best_feature]
        subtree = id3(sub_data, original_data, sub_features, target_name)
        tree[best_feature][val] = subtree

    return tree


def tree_to_dot(tree, parent=None, dot_lines=None, node_id=0):
    if dot_lines is None:
        dot_lines = ["digraph DecisionTree {", "node [shape=box]"]

    if isinstance(tree, dict):
        root = list(tree.keys())[0]
        current_id = node_id
        label = str(root)
        dot_lines.append(f'  node{current_id} [label="{label}"]')
        if parent is not None:
            dot_lines.append(f'  node{parent} -> node{current_id}')
        node_id += 1
        for val, subtree in tree[root].items():
            child_id = node_id
            if isinstance(subtree, dict):
                dot_lines.append(f'  node{current_id} -> node{child_id} [label="{val}"]')
                node_id = tree_to_dot(subtree, current_id, dot_lines, node_id)[1]
            else:
                dot_lines.append(f'  node{child_id} [label="{subtree}", shape=oval]')
                dot_lines.append(f'  node{current_id} -> node{child_id} [label="{val}"]')
                node_id += 1

    if parent is None:
        dot_lines.append('}')
        return "\n".join(dot_lines)
    return "\n".join(dot_lines), node_id


def predict_id3(example, tree):
    if not isinstance(tree, dict):
        return tree
    root = list(tree.keys())[0]
    if root not in example:
        return None
    val = example[root]
    if val not in tree[root]:
        return None
    subtree = tree[root][val]
    return predict_id3(example, subtree)


def run_knn(train_df, test_df, target, k=3, scale=True):
    # Separate X/y
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # One-hot encode categoricals
    X_full = pd.get_dummies(pd.concat([X_train, X_test], ignore_index=True))
    X_train_enc = X_full.iloc[: X_train.shape[0], :]
    X_test_enc = X_full.iloc[X_train.shape[0]:, :]

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_enc = scaler.fit_transform(X_train_enc)
        X_test_enc = scaler.transform(X_test_enc)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_enc, y_train)
    preds = knn.predict(X_test_enc)
    acc = accuracy_score(y_test, preds)
    return knn, scaler, X_train.columns.tolist(), X_full.columns.tolist(), acc


def main():
    st.title("Machine Learning Explorer: ID3 & KNN")
    st.write("Upload a CSV dataset or use the sample dataset, select the target column and algorithm (ID3 or KNN).")

    sample = st.checkbox("Use sample Play Tennis dataset", value=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"]) if not sample else None

    if sample:
        data = pd.read_csv("play_tennis.csv")
    else:
        if uploaded_file is None:
            st.info("Upload a CSV or enable the sample dataset.")
            return
        data = pd.read_csv(uploaded_file)

    st.write("Dataset preview:")
    st.dataframe(data.head())

    cols = list(data.columns)
    target = st.selectbox("Select target (class) column", cols, index=len(cols) - 1)
    features = [c for c in cols if c != target]

    algo = st.selectbox("Algorithm", ["ID3", "KNN"])

    # Allow user to trigger training
    if algo == "ID3":
        # Convert all to strings for categorical handling
        for c in cols:
            data[c] = data[c].astype(str)

        if st.button("Train ID3"):
            train, test = train_test_split(data, test_size=0.3, random_state=42)
            tree = id3(train, train, features, target)

            st.subheader("Learned tree (ID3)")
            dot = tree_to_dot(tree)
            st.graphviz_chart(dot)

            # Evaluate
            y_true = test[target].tolist()
            y_pred = []
            for _, row in test.iterrows():
                ex = row.to_dict()
                y_pred.append(predict_id3(ex, tree))

            maj_class = train[target].mode()[0]
            y_pred = [p if p is not None else maj_class for p in y_pred]
            acc = accuracy_score(y_true, y_pred)
            st.write(f"Accuracy on held-out test set: {acc:.3f}")

            st.subheader("Make a single prediction (ID3)")
            user_input = {}
            for f in features:
                opts = list(data[f].unique())
                user_input[f] = st.selectbox(f, opts, key=f)

            if st.button("Predict ID3"):
                pred = predict_id3(user_input, tree)
                if pred is None:
                    pred = maj_class
                st.write(f"Predicted class: {pred}")

    else:  # KNN
        st.write("KNN settings")
        k = st.number_input("k (neighbors)", min_value=1, max_value=50, value=3)
        scale = st.checkbox("Standard scale numeric features", value=True)

        # Prepare columns: detect numeric vs categorical
        col_types = {}
        for c in features:
            coerced = pd.to_numeric(data[c], errors="coerce")
            if coerced.isna().all():
                col_types[c] = "categorical"
            elif coerced.isna().any():
                col_types[c] = "mixed"
            else:
                col_types[c] = "numeric"

        if st.button("Train KNN"):
            # For KNN, drop rows where target is missing
            knn_data = data.dropna(subset=[target]).copy()
            train, test = train_test_split(knn_data, test_size=0.3, random_state=42)

            knn_model, scaler, orig_cols, enc_cols, acc = run_knn(train, test, target, k=int(k), scale=scale)
            st.write(f"KNN accuracy on held-out test set: {acc:.3f}")

            st.subheader("Make a single prediction (KNN)")
            # Collect a single sample from user using original feature inputs
            sample_input = {}
            for c in features:
                if col_types[c] == "numeric":
                    vals = pd.to_numeric(data[c], errors="coerce")
                    minv = float(vals.min()) if not vals.isna().all() else 0.0
                    maxv = float(vals.max()) if not vals.isna().all() else 1.0
                    meanv = float(vals.median()) if not vals.isna().all() else 0.0
                    sample_input[c] = st.number_input(c, value=meanv, min_value=minv, max_value=maxv)
                else:
                    opts = list(data[c].astype(str).unique())
                    sample_input[c] = st.selectbox(c, opts, key=("knn_" + c))

            if st.button("Predict KNN"):
                # Build one-row DataFrame and encode consistently with training
                X_train = train.drop(columns=[target])
                single = pd.DataFrame([sample_input])
                X_full = pd.get_dummies(pd.concat([X_train, single], ignore_index=True))
                single_enc = X_full.iloc[ X_train.shape[0]: , : ]

                if scale:
                    # Need to refit scaler on train enc; reuse scaler from training if available
                    # Recreate train enc for scaling
                    X_train_full = pd.get_dummies(X_train)
                    scaler_local = StandardScaler()
                    X_train_scaled = scaler_local.fit_transform(X_train_full)
                    # Align single_enc columns with X_train_full columns
                    X_train_full, single_enc = X_train_full.align(single_enc, join="left", axis=1, fill_value=0)
                    single_scaled = scaler_local.transform(single_enc)
                    pred = knn_model.predict(single_scaled)
                else:
                    # Align columns
                    X_train_full = pd.get_dummies(X_train)
                    X_train_full, single_enc = X_train_full.align(single_enc, join="left", axis=1, fill_value=0)
                    pred = knn_model.predict(single_enc)

                st.write(f"Predicted class: {pred[0]}")


if __name__ == "__main__":
    main()
