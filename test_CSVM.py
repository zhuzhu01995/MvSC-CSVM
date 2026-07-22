"""CSVM-MSC GPU test script.

Edit test_list and num_runs below, then run: python test_CSVM.py
"""
import numpy as np
from scipy.io import loadmat
import os
import time
from utils import normalize_data
from algs import alg_scvm_msc
from gpu_utils import to_gpu, get_gpu_info, GPU_AVAILABLE


def main():
    print("=" * 100)
    print("CSVM-MSC GPU-accelerated - Test Script")
    print("=" * 100)
    print("\n" + get_gpu_info())
    print()

    np.random.seed(42)

    data_path = '../MvSC_CSVM_python_GPU/data/'
    if not os.path.exists(data_path):
        data_path = 'data/'

    num_runs = 1

    datasets = [
        {
            'file': 'yale.mat',
            'name': 'Yale',
            'views': 3,
            'lambda': 0.3,
            'gamma': 0.0001,
            'beta': 0.0001,
            'zeta': 0.1,
            'description': '165 samples, 15 classes'
        },
        {
            'file': 'yaleB.mat',
            'name': 'Extended YaleB',
            'views': 3,
            'lambda': 0.002,
            'gamma': 0.00025,
            'beta': 0.0001,
            'zeta': 0.00025,
            'description': '2414 samples, 38 classes'
        },
        {
            'file': 'ORL.mat',
            'name': 'ORL',
            'views': 3,
            'lambda': 0.01,
            'gamma': 0.0001,
            'beta': 0.0001,
            'zeta': 0.01,
            'description': '400 samples, 40 classes'
        },
        {
            'file': 'NH.mat',
            'name': 'Notting Hill',
            'views': 3,
            'lambda': 0.01,
            'gamma': 0.1,
            'beta': 0.0001,
            'zeta': 0.0001,
            'description': '4660 samples, 5 classes'
        },
        {
            'file': 'COIL20MV.mat',
            'name': 'COIL-20',
            'views': 3,
            'lambda': 0.1,
            'gamma': 0.1,
            'beta': 1,
            'zeta': 0.01,
            'description': '1440 samples, 20 classes'
        },
        {
            'file': 'UCI_Digits_X_gt_LogDetFormat.mat',
            'name': 'UCI_Digits',
            'views': 6,
            'lambda': 0.0002,
            'gamma': 0.0002,
            'beta': 0.0001,
            'zeta': 0.001,
            'description': '2000 samples, 10 classes'
        }
    ]

    # 0=Yale, 1=YaleB, 2=ORL, 3=Notting Hill, 4=COIL-20, 5=UCI Digits
    test_list = [5]

    print("=" * 100)
    print("Test configuration:")
    print(f"  Runs: {num_runs}")
    print(f"  Data path: {data_path}")
    print(f"  GPU acceleration: {'enabled' if GPU_AVAILABLE else 'disabled (CPU mode)'}")
    print("\nAvailable datasets:")
    for i, ds in enumerate(datasets):
        marker = "→" if i in test_list else " "
        print(f"  {marker} [{i}] {ds['name']:<20} - {ds['description']:<25} (lambda={ds['lambda']})")
    print(f"\nWill test: {[datasets[i]['name'] for i in test_list]}")
    print("=" * 100)
    print()

    for dataset_idx in test_list:
        ds = datasets[dataset_idx]

        print("\n" + "=" * 100)
        print(f"Testing dataset [{dataset_idx}]: {ds['name']} - {ds['description']}")
        print("=" * 100)

        data_file = os.path.join(data_path, ds['file'])
        if not os.path.exists(data_file):
            print(f'✗ Error: data file {data_file} not found!')
            print(f'  Place .mat files in: {os.path.abspath(data_path)}')
            continue

        print(f"✓ Loaded data: {data_file}")
        data = loadmat(data_file)

        X = []
        for k in range(ds['views']):
            key = f'X{k+1}'
            if key in data:
                X.append(data[key].astype(float))
            else:
                print(f'  ✗ Warning: key {key} not found in data file')

        gt = None
        for label_key in ['gt', 'gnd', 'y']:
            if label_key in data:
                gt = data[label_key].flatten()
                break

        if gt is None:
            print('✗ Error: ground-truth labels not found!')
            continue

        cls_num = len(np.unique(gt))
        K = len(X)

        print(f"✓ Data loaded successfully:")
        print(f"  Views: {K}")
        print(f"  Samples: {X[0].shape[1]}")
        print(f"  Feature dims: {[x.shape[0] for x in X]}")
        print(f"  Classes: {cls_num}")

        print(f"✓ Normalizing data{' and transferring to GPU' if GPU_AVAILABLE else ''}...")
        Y = [to_gpu(normalize_data(X[iv])) for iv in range(K)]

        opts = {
            'Frac_alpha': 5000,
            'maxIter': ds.get('maxIter', 60),
            'epsilon': ds.get('epsilon', 1e-4),
            'flag_debug': 0,
            'mu': 1e-5,
            'eta': 2,
            'max_mu': 1e10,
            'lambda': ds['lambda'],
        }
        for _k in ('gamma', 'beta', 'zeta'):
            if _k in ds:
                opts[_k] = ds[_k]

        _g = opts.get('gamma', '(default)')
        _b = opts.get('beta', '(default)')
        _z = opts.get('zeta', '(default)')
        print(
            f"✓ Algorithm params: lambda={ds['lambda']}, gamma={_g}, beta={_b}, zeta={_z}, "
            f"maxIter={opts['maxIter']}, epsilon={opts['epsilon']}"
        )

        run_times = []
        NMI_results = []
        AR_results = []
        ACC_results = []
        recall_results = []
        precision_results = []
        fscore_results = []

        print(f"\nRunning algorithm ({num_runs} run(s)):")
        print("-" * 100)

        total_start = time.time()

        for run_idx in range(num_runs):
            print(f"  [{run_idx+1:2d}/{num_runs}] ", end='', flush=True)

            run_start = time.time()
            C, C_Fusion, Out = alg_scvm_msc(Y, cls_num, gt, opts)
            elapsed = time.time() - run_start

            run_times.append(elapsed)
            NMI_results.append(Out['NMI'])
            AR_results.append(Out['AR'])
            ACC_results.append(Out['ACC'])
            recall_results.append(Out['recall'])
            precision_results.append(Out['precision'])
            fscore_results.append(Out['fscore'])

            print(f"Time: {elapsed:6.2f}s  |  "
                  f"NMI: {Out['NMI']:.4f}  "
                  f"ACC: {Out['ACC']:.4f}  "
                  f"AR: {Out['AR']:.4f}  "
                  f"F: {Out['fscore']:.4f}  "
                  f"Precision: {Out['precision']:.4f}  "
                  f"Recall: {Out['recall']:.4f}")

        total_time = time.time() - total_start

        print("-" * 100)
        print("\nResults summary:")
        print("=" * 100)
        print(f"{'Metric':<12}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
        print("-" * 100)

        metrics = {
            'Time (s)': run_times,
            'NMI': NMI_results,
            'ACC': ACC_results,
            'AR': AR_results,
            'Recall': recall_results,
            'Precision': precision_results,
            'F-score': fscore_results,
        }

        for name, values in metrics.items():
            print(f"{name:<12}  {np.mean(values):>10.4f}  {np.std(values):>10.4f}  "
                  f"{np.min(values):>10.4f}  {np.max(values):>10.4f}")

        print("=" * 100)
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Mean time per run: {np.mean(run_times):.2f}s")
        print(f"GPU acceleration: {'enabled' if GPU_AVAILABLE else 'disabled'}")

        best_idx = np.argmax(ACC_results)
        print(f"\nBest run (run #{best_idx+1}):")
        print(f"  ACC: {ACC_results[best_idx]:.4f}")
        print(f"  NMI: {NMI_results[best_idx]:.4f}")
        print(f"  F-score: {fscore_results[best_idx]:.4f}")
        print(f"  Time: {run_times[best_idx]:.2f}s")

    print("\n" + "=" * 100)
    print("All tests completed!")
    print("=" * 100)


if __name__ == '__main__':
    main()
