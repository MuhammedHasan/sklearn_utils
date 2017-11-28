from sklearn.pipeline import Pipeline


class DynamicPreprocessing:
    '''
    Dynamic Preprocessing
    '''

    def __new__(cls, selected_steps=None):
        selected_steps = selected_steps or self.default_steps
        return Pipeline([(i, self.steps[i]) for i in selected_steps])
