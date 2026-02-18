#!/usr/bin/env python3
"""Train/Test 최적화 실시간 모니터링"""
import optuna
from datetime import datetime
import time
import os

def monitor():
    while True:
        os.system('clear')

        study = optuna.load_study(
            study_name='ptj_v3_train_test_v2',
            storage='sqlite:///data/optuna_v3_train_test_v2.db'
        )
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print('=' * 80)
        print(f'  PTJ v3 Train/Test 최적화 실시간 모니터링')
        print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 80)
        print()
        print(f'진행률: {len(completed)}/300 trials ({len(completed)/300*100:.1f}%)')
        print()

        if len(completed) > 0:
            # 최근 15개 trial
            recent = sorted(completed, key=lambda t: t.number, reverse=True)[:15]
            print('최근 15개 Trial:')
            print('-' * 80)
            print(f'{"#":>6} | {"Train":>8} | {"Test":>8} | {"차이":>8} | {"평가":>10}')
            print('-' * 80)
            for t in reversed(recent):
                train = t.user_attrs.get('train_return', t.value)
                test = t.user_attrs.get('test_return', 0)
                diff = t.user_attrs.get('degradation', 0)

                # 과최적화 평가
                if diff < 2:
                    status = '우수'
                elif diff < 5:
                    status = '주의'
                else:
                    status = '과최적'

                print(f'{t.number:6d} | {train:+7.2f}% | {test:+7.2f}% | {diff:+7.2f}%p | {status:>10}')

            print()
            print('=' * 80)
            # Best trial
            best = study.best_trial
            best_train = best.user_attrs.get('train_return', best.value)
            best_test = best.user_attrs.get('test_return', 0)
            best_diff = best.user_attrs.get('degradation', 0)

            print(f'현재 Best: Trial #{best.number}')
            print(f'   Train: {best_train:+7.2f}% | Test: {best_test:+7.2f}% | 차이: {best_diff:+7.2f}%p')

            # 과최적화 평가
            if best_diff < 2:
                print(f'   평가: 강건한 전략 (성능 차이 2% 미만)')
            elif best_diff < 5:
                print(f'   평가: 모니터링 필요 (성능 차이 2-5%)')
            else:
                print(f'   평가: 과최적화 위험 (성능 차이 5% 이상)')

            # 통계
            avg_train = sum(t.user_attrs.get('train_return', t.value) for t in completed) / len(completed)
            avg_test = sum(t.user_attrs.get('test_return', 0) for t in completed) / len(completed)
            avg_diff = sum(t.user_attrs.get('degradation', 0) for t in completed) / len(completed)

            print()
            print(f'평균 성능:')
            print(f'   Train: {avg_train:+7.2f}% | Test: {avg_test:+7.2f}% | 차이: {avg_diff:+7.2f}%p')

        print('=' * 80)
        print('다음 업데이트: 60초 후... (Ctrl+C로 종료)')
        print('=' * 80)

        time.sleep(60)

if __name__ == "__main__":
    monitor()
