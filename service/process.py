import os
import joblib
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_model_and_params(model_name, depth):
    if model_name.lower() == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [depth, depth + 2, depth + 4],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name.lower() == 'xgb':
        model = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            min_split_gain=0.0,
            min_child_samples=10
        )
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001]
        }
    elif model_name.lower() == 'lgbm':
        model = LGBMClassifier(eval_metric='logloss', random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [depth, depth + 2, depth + 4],
            'learning_rate': [0.1, 0.01, 0.001]
        }
    else:
        raise ValueError("모델 타입은 'rf', 'xgb', 'lgbm' 중 하나여야 합니다.")
    return model, param_grid

def train(args, data, label):
    """
    1) KFold 교차 검증으로 하이퍼파라미터 튜닝 & 성능 평가 (train+valid 데이터)
    2) (옵션) --debug_split을 통해 fold별 train/valid 분포 확인
    """
    kfold = KFold(n_splits=args.split_num, shuffle=True, random_state=42)
    model, param_grid = get_model_and_params(args.model_type, args.depth)

    fold_num = 0
    acc_list = []
    f1_list = []
    best_params_list = []  # 각 fold의 최적 하이퍼파라미터 저장

    for train_index, valid_index in kfold.split(data):
        fold_num += 1
        X_train, X_valid = data.iloc[train_index], data.iloc[valid_index]
        y_train, y_valid = label.iloc[train_index], label.iloc[valid_index]

        # 디버그 옵션: fold별 레이블 분포 출력
        if args.debug_split:
            print(f"\n[Fold {fold_num}]")
            print("  Train label distribution:\n", y_train.value_counts())
            print("  Valid label distribution:\n", y_valid.value_counts())

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params_list.append(grid_search.best_params_)

        pred = best_model.predict(X_valid)
        accuracy = np.round(accuracy_score(y_valid, pred), 4)
        f1score = np.round(f1_score(y_valid, pred), 4)
        precision = np.round(precision_score(y_valid, pred), 4)
        recall = np.round(recall_score(y_valid, pred), 4)

        print(f"\n#{fold_num} 교차 검증 결과")
        print(f"  - Best Params: {grid_search.best_params_}")
        print(f"  - Accuracy: {accuracy}, F1: {f1score}, Precision: {precision}, Recall: {recall}")
        print(f"  - 학습 데이터 크기: {X_train.shape[0]}, 검증 데이터 크기: {X_valid.shape[0]}")

        acc_list.append(accuracy)
        f1_list.append(f1score)

    print("\n## 평균 검증 Accuracy:", np.mean(acc_list))
    print("## 평균 검증 F1-score:", np.mean(f1_list))

    return best_params_list

# def final_evaluation(args, train_valid_data, train_valid_label, test_data, test_label, best_params_list):
#     final_params = best_params_list[-1]
#     model, _ = get_model_and_params(args.model_type, args.depth)
    
#     if args.model_type.lower() == 'xgb':
#         final_model = XGBClassifier(**final_params, eval_metric='logloss', random_state=42, min_split_gain=0.0, min_child_samples=10)
#     elif args.model_type.lower() == 'lgbm':
#         final_model = LGBMClassifier(**final_params, eval_metric='logloss', random_state=42)
#     else:
#         final_model = RandomForestClassifier(**final_params, random_state=42)
    
#     final_model.fit(train_valid_data, train_valid_label)
#     pred_test = final_model.predict(test_data)
#     acc_test = accuracy_score(test_label, pred_test)
#     f1_test = f1_score(test_label, pred_test)
#     precision_test = precision_score(test_label, pred_test)
#     recall_test = recall_score(test_label, pred_test)
    
#     print("\n## 최종 테스트 평가 결과")
#     print(f"  - Accuracy: {acc_test}")
#     print(f"  - F1-score: {f1_test}")
#     print(f"  - Precision: {precision_test}")
#     print(f"  - Recall: {recall_test}")

#     # 모델 저장
#     os.makedirs("model", exist_ok=True)
#     model_path = os.path.join("model", f"final_model_{args.model_type}.pkl")
#     joblib.dump(final_model, model_path)
#     print(f"✅ 모델 저장 완료: {model_path}")

def final_evaluation(args, train_valid_data, train_valid_label, test_data, test_label, best_params, cluster_num):
    final_params = best_params[-1]
    model, _ = get_model_and_params(args.model_type, args.depth)
    
    if args.model_type.lower() == 'xgb':
        final_model = XGBClassifier(**final_params, eval_metric='logloss', random_state=42, min_split_gain=0.0, min_child_samples=10)
    elif args.model_type.lower() == 'lgbm':
        final_model = LGBMClassifier(**final_params, eval_metric='logloss', random_state=42)
    else:
        final_model = RandomForestClassifier(**final_params, random_state=42)
    
    final_model.fit(train_valid_data, train_valid_label)
    pred_test = final_model.predict(test_data)
    acc_test = accuracy_score(test_label, pred_test)
    f1_test = f1_score(test_label, pred_test)
    precision_test = precision_score(test_label, pred_test)
    recall_test = recall_score(test_label, pred_test)
    
    print(f"\n## 클러스터 {cluster_num} 최종 테스트 평가 결과")
    print(f"  - Accuracy: {acc_test}")
    print(f"  - F1-score: {f1_test}")
    print(f"  - Precision: {precision_test}")
    print(f"  - Recall: {recall_test}")

    # 모델 저장 (클러스터별 모델)
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", f"Cluster{cluster_num}_model_{args.model_type}.pkl")
    joblib.dump(final_model, model_path)
    print(f"✅ 클러스터 {cluster_num} 모델 저장 완료: {model_path}")


# def run_full_process(args):
#     print("🚀 데이터 로드 중...")
#     df = pd.read_csv("data/clustered_Customer_Churn.csv")
#     X = df.drop('Exited', axis=1)
#     y = df['Exited']
#     print("✅ 데이터 로드 완료")
    
#     print("🚀 모델 학습 시작...")
#     best_params = train(args, X, y)
#     print("✅ 모델 학습 완료")
    
#     print("🚀 최종 모델 평가...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     final_evaluation(args, X_train, y_train, X_test, y_test, best_params)
    
#     return best_params

def run_full_process(args):
    print("🚀 데이터 로드 중...")

    best_params_dict = {}  # 클러스터별 최적 하이퍼파라미터 저장
    
    for cluster_num in [0, 3]:
        file_path = f"data/Cluster_{cluster_num}_Preprocessed.csv"
        
        if not os.path.exists(file_path):
            print(f"❌ 파일 없음: {file_path}, 클러스터 {cluster_num} 데이터가 존재하지 않습니다.")
            continue
        
        print(f"\n📌 클러스터 {cluster_num} 데이터 로드 중...")
        df = pd.read_csv(file_path)
        
        if "Exited" not in df.columns:
            print(f"❌ 'Exited' 컬럼 없음: {file_path}")
            continue
        
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        print(f"✅ 클러스터 {cluster_num} 데이터 로드 완료 (샘플 수: {X.shape[0]})")
        
        print(f"🚀 클러스터 {cluster_num} 모델 학습 시작...")
        best_params = train(args, X, y)
        best_params_dict[cluster_num] = best_params
        print(f"✅ 클러스터 {cluster_num} 모델 학습 완료")
        
        print(f"🚀 클러스터 {cluster_num} 최종 모델 평가...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        final_evaluation(args, X_train, y_train, X_test, y_test, best_params, cluster_num)
    
    return best_params_dict
