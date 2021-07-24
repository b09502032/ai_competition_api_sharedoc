# 你，識字嗎？ api server 架構分享

這是我們這隊玉山人工智慧挑戰賽2021夏季賽【:dizzy:最佳API服務獎】的說明文件。

## 架構

#### 機器規格

| key          | value                                 |
| ------------ | ------------------------------------- |
| Machine type | e2-highcpu-16 (16 vCPU, 16 GB memory) |
| Zone         | asia-east1-b                          |
| Image        | Debian GNU/Linux 10 (buster)          |

#### 系統架構圖

![structure](structure.svg)

## Usage

individual node:

```
usage: api.py [-h] --primary PRIMARY [PRIMARY ...] [--primary-threshold PRIMARY_THRESHOLD]
              [--prelim PRELIM [PRELIM ...]] [--prelim-threshold PRELIM_THRESHOLD]
              [--max-workers MAX_WORKERS] [--training-data-dic TRAINING_DATA_DIC] [--data DATA]
              [--captain-email CAPTAIN_EMAIL] [--salt SALT]

optional arguments:
  -h, --help            show this help message and exit
  --primary PRIMARY [PRIMARY ...]
                        primary checkpoints to ensemble (default: None)
  --primary-threshold PRIMARY_THRESHOLD
                        (default: 0.28)
  --prelim PRELIM [PRELIM ...]
                        prelim checkpoints to ensemble (default: None)
  --prelim-threshold PRELIM_THRESHOLD
                        (default: 0.7)
  --max-workers MAX_WORKERS
                        The maximum number of processes that can be used to execute the predict
                        function calls (default: 1)
  --training-data-dic TRAINING_DATA_DIC
  --data DATA           directory to save requests and responses (default: data)
  --captain-email CAPTAIN_EMAIL
  --salt SALT
```

If `--training-data-dic /path/to/training data dic.txt` is specified, predictions not in `training data dic.txt` will be converted to "isnull".

Additional arguments will be passed as gunicorn settings (bind, threads, timeout, etc.).

If not specified, captain email, salt, and other gunicorn settings will be loaded from [`config.py`](src/config.py).

Example:

```bash
python3 --primary inception_v3_primary/models/best_macro_average_f1.ckpt dm_nfnet_f0_primary/models/best_macro_average_f1.ckpt repvgg_b3g4_primary/models/best_macro_average_f1.ckpt resnetv2_101x1_bitm_primary/models/best_macro_average_f1.ckpt --prelim repvgg_b3g4_prelim/models/best_macro_average_f1.ckpt --training-data-dic 'training data dic.txt' --bind '0.0.0.0:24865'
```

load balancer:

```
usage: balance.py [-h] [--netloc NETLOC [NETLOC ...]] [--max-workers MAX_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --netloc NETLOC [NETLOC ...]
                        netloc of api server node (default: None)
  --max-workers MAX_WORKERS
```

If `--max-workers` is not specified, it will be 4 * number_of_netlocs.

Additional arguments will be passed as gunicorn settings (bind, threads, timeout, etc.).

If not specified, netlocs and other gunicorn settings will be loaded from [`config.py`](src/config.py).

Example:

```bash
python3 balance.py --netloc ADDRESS1 ADDRESS2 --bind '0.0.0.0:44966' --threads 120
```
