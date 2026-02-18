#!/usr/bin/env python3
"""Optuna study 이름 확인"""
import optuna

# 모든 study 확인
storage = "sqlite:///optuna_v3_train_test.db"
studies = optuna.study.get_all_study_summaries(storage=storage)

print(f"Found {len(studies)} studies:")
for s in studies:
    print(f"  - {s.study_name} (trials: {s.n_trials})")
