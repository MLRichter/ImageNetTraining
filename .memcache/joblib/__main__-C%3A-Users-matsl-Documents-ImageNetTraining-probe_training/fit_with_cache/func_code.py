# first line: 149
@memory.cache
def fit_with_cache(data: np.ndarray, labels: np.ndarray, verbose: int = 0):
    print("Start training with", data.shape, labels.shape)
    model = LogisticRegressionModel(
        multi_class='multinomial', n_jobs=12, solver='saga', verbose=verbose
    ).fit(data, labels)
    return model
