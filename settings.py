OGB_ROOT = './dataset/'
DATA_PATH = 'data/'

DATA_INFO = {
    'arxiv': {
        'type': 'ogb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 40,
        'n_nodes': 169343,
        'ogb_name': 'ogbn-arxiv',
        'raw_data_path': OGB_ROOT,  # Place to save raw data
        'max_length': 512,  # Place to save raw data
        'data_root': f'{OGB_ROOT}ogbn_arxiv',  # Default ogb download target path
        'raw_text_url': 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz',
    },
    'papers100M': {
        'type': 'ogb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 172,
        'n_nodes': 111059956, #110479148
        'ogb_name': 'ogbn-papers100M',  #
        'download_name': 'paperinfo',
        'raw_data_path': OGB_ROOT,  # Place to save raw data
        'data_root': f'{OGB_ROOT}ogbn_papers100M/',  # Default ogb download target path
        'max_length': 129,
        'raw_text_url': 'https://snap.stanford.edu/ogb/data/misc/ogbn_papers100M/paperinfo.zip',
    },
    'products': (product_settings := {
        'type': 'ogb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 47,
        'n_nodes': 2449029,
        'max_length': 512,
        'ogb_name': 'ogbn-products',  #
        'download_name': 'AmazonTitles-3M',
        'raw_data_path': OGB_ROOT,  # Place to save raw data
        'data_root': f'{OGB_ROOT}ogbn_products/',  # Default ogb download target path
        'raw_text_url': 'https://drive.google.com/u/0/uc?id=1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN&export=download'
    }),
    'products256': {
        **product_settings,
        'cut_off': 256
    },
    'cora_ml': {
        'train_ratio': 0.5,
        'valid_ratio': 0.1, 
        'test_ratio': 0.4,
        'n_labels': 7,
        'n_nodes': 2277,
        'max_length': 512,
        'ogb_name': 'cora_ml',
        'raw_data_path': OGB_ROOT,
        'data_root': f'{OGB_ROOT}cora_ml/'
    },
    'wikics': {
        'type': 'wikics',
        'n_labels': 10,
        'n_nodes': 11701,
        'max_length': 512,
        'ogb_name': 'wikics'
    },
    'FB15K237': {
        'type': 'FB15K237',
        'n_labels': 237,
        'n_nodes': 14541,
        'max_length': 512,
        'ogb_name': 'FB15K237'
    },
    'WN18RR': {
        'type': 'WN18RR',
        'n_labels': 11,
        'n_nodes': 40943,
        'max_length': 512,
        'ogb_name': 'WN18RR'
    },
    'cora': {
        'type': 'cora',
        'n_labels': 7,
        'n_nodes': 2708,
        'max_length': 512,
        'ogb_name': 'cora'
    },
    'pubmed': {
        'type': 'pubmed',
        'n_labels': 3,
        'n_nodes': 19717,
        'max_length': 512,
        'ogb_name': 'pubmed'
    },
    'hiv': {
        'type': 'hiv',
        'n_labels': 1,
        'n_nodes': 41127,
        'max_length': 512,
        'ogb_name': 'hiv'
    },
    'pcba': {
        'type': 'pcba',
        'n_labels': 128,
        'n_nodes': 437929,
        'max_length': 512,
        'ogb_name': 'pcba'
    },
}