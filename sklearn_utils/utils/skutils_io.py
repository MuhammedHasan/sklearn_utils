class SkUtilsIO:
    '''
    IO class to read and write dataset to file.
    '''

    def __init__(filename, gz=False):
        '''
        :filename: file name with path.
        '''
        self.filename = filename

    def from_csv():
        pass

    def to_csv(X, y):
        pass

    def from_json():
        pass

    def to_json(X, y):
        '''
        :X: dataset as list of dict.
        :y: labels.
        '''
        with gzip.open('%s.gz' % self.filename, 'wt') if self.gz else open(
                self.path, 'w') as file:
            json.dump(list(zip(X, y)), file)
