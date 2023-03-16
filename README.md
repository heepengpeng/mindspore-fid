# FID score for MindSpore

## Usage

### Compute the FID score between two datasets

```python
python fid_score.py --sample_data_path path/to/sampledataset --ref_data_path path/to/refdataset
```

### Generating a compatible .npz archive from a dataset
```python
python fid_score.py --sample_data_path path/to/sampledataset --ref_data_path path/to/outputfile --save_stats true
```

### Compute the FID score between .npz 
```python
python fid_score.py --sample_data_path path/to/sample.npz --ref_data_path path/to/ref.npz
```