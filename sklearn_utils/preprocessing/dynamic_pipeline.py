from sklearn.pipeline import Pipeline


class DynamicPipeline:
    '''
    Dynamic Pipeline
    '''

    def __new__(cls, selected_steps=None):
        selected_steps = selected_steps or cls.default_steps
        return cls.make_pipeline(selected_steps)

    @classmethod
    def make_pipeline(cls, selected_steps):
        return Pipeline([(i, cls.steps[i]) for i in selected_steps])
