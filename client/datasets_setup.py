from pathlib import Path

import requests

DATA_DIR = Path('./data/')
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['a4a', 'a9a']
DATASETS_EXTRA = ['square', 'flipcross', 'cross', 'adult', 'credit', 'recidivism',  'lending', 'heloc']

DATASET_PATHS = {ds: {'train': DATA_DIR / ds, 'test': DATA_DIR / f'{ds}.t'} for ds in DATASETS}

DATASET_DEFAULT_PATHS = {ds: {'train': DATA_DIR / ds / f'{ds}_train.txt', 'test': DATA_DIR / ds / f'{ds}_test.txt'} for ds in DATASETS_EXTRA}
DATASET_BASE_PATHS = {ds: {'train': DATA_DIR / ds / f'{ds}_train_base.txt', 'test': DATA_DIR / ds / f'{ds}_test_base.txt'} for ds in DATASETS_EXTRA}
DATASET_FLIP_PATHS = {ds: {'train': DATA_DIR / ds / f'{ds}_train_flip.txt', 'test': DATA_DIR / ds / f'{ds}_test_flip.txt'} for ds in DATASETS_EXTRA}
DATASET_CORR_PATHS = {ds: {'train': DATA_DIR / ds / f'{ds}_train_corr.txt', 'test': DATA_DIR / ds / f'{ds}_test_corr.txt'} for ds in DATASETS_EXTRA}
DATASET_EXPAND_PATHS = {ds: {'train': DATA_DIR / ds / f'{ds}_train_expand.txt', 'test': DATA_DIR / ds / f'{ds}_test_expand.txt'} for ds in DATASETS_EXTRA}


def url_retrieve(url: str, outfile: Path):
  R = requests.get(url, allow_redirects=True)
  if R.status_code != 200:
    raise ConnectionError('could not download {}\nerror code: {}'.format(url, R.status_code))

  outfile.write_bytes(R.content)


def download_UCI_dataset(ds):
  print(f'Downloading libsvm {ds}-train (derived from UCI Adult dataset)...')
  url_retrieve(f'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{ds}', DATASET_PATHS[ds]['train'])
  print(f'Downloading libsvm {ds}-test  (derived from UCI Adult dataset)...')
  url_retrieve(f'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{ds}.t', DATASET_PATHS[ds]['test'])


if __name__ == '__main__':
  for ds in DATASETS:
    download_UCI_dataset(ds)
