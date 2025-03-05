from service.process import run_full_process
from service.preprocess import run_preprocessing
import argparse
import os

def main():
    print("🚀 고객 이탈 예측 데이터 전처리 시작...")
    
    parser = argparse.ArgumentParser(description="데이터 전처리 및 모델 학습 실행")
    parser.add_argument("--cluster_num", type=int, choices=[0, 3], default=0, help="클러스터 번호 (기본값: 0)")
    parser.add_argument("--split_num", type=int, default=5, help="KFold 교차검증 분할 수")
    parser.add_argument("--depth", type=int, default=2, help="모델의 기본 max_depth 값")
    parser.add_argument("--drop_col", nargs='+', default=["CustomerId", "Surname"], help="삭제할 컬럼 리스트")
    parser.add_argument("--onehot_col", nargs='+', default=["Geography", "Gender"], help="One-Hot 인코딩할 컬럼 리스트")
    parser.add_argument("--ordi_col", default="Card Type", help="Ordinal 인코딩할 컬럼")
    parser.add_argument("--model_type", choices=['rf', 'xgb', 'lgbm'], default="rf", help="사용할 모델 타입: rf, xgb, lgbm")
    parser.add_argument("--debug_split", action='store_true', help="교차 검증 시 fold별 데이터 split 정보를 출력")
    parser.add_argument("--plot_loss", action='store_true', help="학습 시 손실 곡선을 시각화할지 여부")
    
    args = parser.parse_args()
    
    print(f"✅ 선택한 클러스터 번호: {args.cluster_num}")
    df = run_preprocessing(args)    
    print("✅ 전체 전처리 완료!")
    
    print("🚀 모델 학습 시작...")
    best_params = run_full_process(args)
    print("✅ 모델 학습 완료! 최적 하이퍼파라미터:", best_params)

if __name__ == "__main__":
    main()
