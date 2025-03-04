import numpy as np
import pandas as pd
import os
import argparse
from category_encoders import OrdinalEncoder, OneHotEncoder

# from category_encoders import OrdinalEncoder, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import KFold, GridSearchCV, train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# import xgboost as xgb


###########################################################################
#
###########################################################################


###########################################################################
# 충성도 피쳐쳐 추가, 클러스터 0, 3분리후 데이터 셋 저장
###########################################################################


def load_data(file_name="Customer-Churn-Records.csv"):
    """ data 폴더에서 CSV 파일을 불러오는 함수 """
    file_path = os.path.join("data", file_name)
    return pd.read_csv(file_path)

def add_loyalty_score(df):
    """ 고객 충성도 점수 (Loyalty_Score) 추가 """
    df["Loyalty_Score"] = (
        df["IsActiveMember"] * 2 +  # 활성 회원 여부 (가중치 2)
        df["Tenure"] * 1.5 +  # 가입 기간 (가중치 1.5)
        df["NumOfProducts"] * 2 +  # 가입 상품 수 (가중치 2)
        df["Satisfaction Score"] * 3 +  # 고객 만족도 (가중치 3)
        df["Point Earned"] / 100  # 포인트 적립 (스케일 조정)
    )
    return df

###########################################################################
# 데이터 전처리 실행 함수
###########################################################################

def process_clusters(data_path, df):
    """ 클러스터 0과 3을 필터링하고 저장하는 함수 """
    os.makedirs(data_path, exist_ok=True)

    file_cluster_0 = os.path.join(data_path, "Cluster_0_Filtered.csv")
    file_cluster_3 = os.path.join(data_path, "Cluster_3_Filtered.csv")

    df_cluster_0 = df[(df["Age"] >= 44) & (df["EstimatedSalary"] >= 100000) & (df["NumOfProducts"] >= 1)]
    df_cluster_3 = df[(df["Age"] <= 40) & (df["EstimatedSalary"] >= 70000) & (df["Balance"] >= 90000) & (df["Loyalty_Score"] >= 26)]
    
    df_cluster_0.to_csv(file_cluster_0, index=False)
    df_cluster_3.to_csv(file_cluster_3, index=False)
    
    print(f"📂 클러스터 0 데이터 저장 완료: {file_cluster_0}")
    print(f"📂 클러스터 3 데이터 저장 완료: {file_cluster_3}")


###########################################################################
# 
###########################################################################

def compute_complain_score_cluster_0(dataframe):
    """ 클러스터 0: 불만(Complain) 점수 계산 """
    dataframe['z_Age'] = (dataframe['Age'] - dataframe['Age'].mean()) / dataframe['Age'].std()
    dataframe['z_EstimatedSalary'] = (dataframe['EstimatedSalary'] - dataframe['EstimatedSalary'].mean()) / dataframe['EstimatedSalary'].std()
    dataframe['z_NumOfProducts'] = (dataframe['NumOfProducts'] - dataframe['NumOfProducts'].mean()) / dataframe['NumOfProducts'].std()
    
    corr = dataframe[['Age', 'EstimatedSalary', 'NumOfProducts', 'Complain']].corr()
    w_age = corr.loc['Complain', 'Age'] / corr.loc['Complain'].abs().sum() - 1
    w_est = corr.loc['Complain', 'EstimatedSalary'] / corr.loc['Complain'].abs().sum() - 1
    w_num = corr.loc['Complain', 'NumOfProducts'] / corr.loc['Complain'].abs().sum() - 1
    
    dataframe['Complain_score'] = (5e-5 + dataframe['Complain']) * (
        w_age * dataframe['z_Age'] + w_est * dataframe['z_EstimatedSalary'] + w_num * dataframe['z_NumOfProducts']
    )
    dataframe.drop(['z_Age', 'z_EstimatedSalary', 'z_NumOfProducts', 'Complain'], axis=1, inplace=True)
    return dataframe

def compute_complain_score_cluster_3(dataframe):
    """ 클러스터 3: 불만(Complain) 점수 계산 """
    dataframe['z_Age'] = (dataframe['Age'] - dataframe['Age'].mean()) / dataframe['Age'].std()
    dataframe['z_EstimatedSalary'] = (dataframe['EstimatedSalary'] - dataframe['EstimatedSalary'].mean()) / dataframe['EstimatedSalary'].std()
    dataframe['z_Balance'] = (dataframe['Balance'] - dataframe['Balance'].mean()) / dataframe['Balance'].std()
    dataframe['z_Loyalty_Score'] = (dataframe['Loyalty_Score'] - dataframe['Loyalty_Score'].mean()) / dataframe['Loyalty_Score'].std()
    
    corr = dataframe[['Age', 'EstimatedSalary', 'Balance', 'Loyalty_Score', 'Complain']].corr()
    w_age = corr['Complain']['Age'] / corr['Complain'].abs().sum()-1
    w_est = corr['Complain']['EstimatedSalary'] / corr['Complain'].abs().sum()-1
    w_bal = corr['Complain']['Balance'] / corr['Complain'].abs().sum()-1
    w_lcs = corr['Complain']['Loyalty_Score'] / corr['Complain'].abs().sum()-1
    
    dataframe['Complain_score'] = dataframe['Complain'] * (
        w_age * dataframe['z_Age'] + w_est * dataframe['z_EstimatedSalary'] + w_bal * dataframe['z_Balance'] + w_lcs * dataframe['z_Loyalty_Score']
    )
    dataframe.drop(['z_Age', 'z_EstimatedSalary', 'z_Balance', 'z_Loyalty_Score', 'Complain'], axis=1, inplace=True)
    return dataframe



def preprocess(args, dataframe: pd.DataFrame):
    """ 전처리 수행 (인코딩, 불필요한 컬럼 삭제, 가중치 계산) """
    dataframe.drop(args.drop_col, axis=1, inplace=True, errors='ignore')
    
    if set(args.onehot_col).issubset(dataframe.columns):
        OHE = OneHotEncoder(cols=args.onehot_col, use_cat_names=True)
        dataframe = OHE.fit_transform(dataframe)
        dataframe.drop(args.onehot_col, axis=1, inplace=True, errors='ignore')
    
    if isinstance(args.ordi_col, str):
        ordi_cols = [args.ordi_col]
    else:
        ordi_cols = args.ordi_col
    
    if set(ordi_cols).issubset(dataframe.columns):
        ODE = OrdinalEncoder(cols=ordi_cols)
        dataframe = ODE.fit_transform(dataframe)
        dataframe.drop(ordi_cols, axis=1, inplace=True, errors='ignore')
    
    if args.cluster_num == 0:
        dataframe = compute_complain_score_cluster_0(dataframe)
    elif args.cluster_num == 3:
        dataframe = compute_complain_score_cluster_3(dataframe)
    
    X = dataframe.drop('Exited', axis=1)
    y = dataframe['Exited']
    return X, y


###########################################################################
# 실행 파일
###########################################################################

###########################################################################
# 실행 파일 (수정된 run_preprocessing)
###########################################################################

def run_preprocessing(args):
    """ 전처리 실행 함수 (클러스터 0과 3 모두 적용) """
    print("🚀 데이터 로드 시작...")
    df = load_data()
    print("✅ 원본 데이터 로드 완료")
    
    print("🚀 충성도 점수 추가...")
    df = add_loyalty_score(df)
    print("✅ 충성도 점수 추가 완료")
    
    print("🚀 클러스터링 데이터 저장...")
    process_clusters("data", df)
    print("✅ 클러스터링 데이터 저장 완료")
    
    # 📌 클러스터 0과 3을 모두 처리
    processed_data = {}
    for cluster_num in [0, 3]:
        print(f"🚀 클러스터 {cluster_num} 데이터 로드...")
        file_name = f"Cluster_{cluster_num}_Filtered.csv"
        
        if not os.path.exists(os.path.join("data", file_name)):
            print(f"❌ 클러스터 {cluster_num} 데이터 파일이 존재하지 않습니다: {file_name}")
            continue
        
        df_cluster = load_data(file_name)
        print(f"✅ 클러스터 {cluster_num} 데이터 로드 완료 (샘플 수: {df_cluster.shape[0]})")
        
        print(f"🚀 클러스터 {cluster_num} 데이터 전처리 시작...")
        X, y = preprocess(args, df_cluster)
        print(f"✅ 클러스터 {cluster_num} 데이터 전처리 완료")
        
        # 📌 전처리된 데이터를 'data' 폴더에 저장
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)  # 폴더가 없으면 생성

        processed_file = os.path.join(data_dir, f"Cluster_{cluster_num}_Preprocessed.csv")
        X["Exited"] = y  # 정답값(이탈 여부) 추가
        X.to_csv(processed_file, index=False)
        
        print(f"✅ 클러스터 {cluster_num} 전처리된 데이터 저장 완료: {processed_file}")
        
        # 결과 저장 (모델 학습 시 사용할 수 있도록 반환)
        processed_data[cluster_num] = (X, y)
    
    print("✅ 최종 전처리 완료!")
    return processed_data


# def run_preprocessing(args):
#     """ 전처리 실행 함수 """
#     print("🚀 데이터 로드 시작...")
#     df = load_data()
#     print("✅ 원본 데이터 로드 완료")
    
#     print("🚀 충성도 점수 추가...")
#     df = add_loyalty_score(df)
#     print("✅ 충성도 점수 추가 완료")
    
#     print("🚀 클러스터링 데이터 저장...")
#     process_clusters("data", df)
#     print("✅ 클러스터링 데이터 저장 완료")
    
#     print(f"🚀 클러스터 {args.cluster_num} 데이터 로드...")
#     file_name = f"Cluster_{args.cluster_num}_Filtered.csv"
#     df = load_data(file_name)
#     print("✅ 클러스터 데이터 로드 완료")
    
#     print("🚀 데이터 전처리 시작...")
#     X, y = preprocess(args, df)
#     print("✅ 데이터 전처리 완료")
#     print("✅ 최종 전처리 완료!")
    
#     return X, y