import concurrent.futures
import functools
import typing

import numpy
import PIL.Image
import torch

import models
import transforms


class Classifier:
    def __init__(
        self,
        model_name: str = None,
        classes: typing.List[str] = None,
        transform_name: str = None,
        output_to_idx: typing.Callable[[torch.Tensor], torch.Tensor] = None,
        pretained: bool = True,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.classes = classes
        self.transform_name = transform_name
        self.output_to_idx = output_to_idx
        if model_name is not None:
            self.model: torch.nn.Module = models.get_model(model_name, len(classes), pretrained=pretained, **kwargs)
        if transform_name is not None:
            self.transform: typing.Callable[[typing.Union[PIL.Image.Image, numpy.ndarray]], torch.Tensor] = getattr(transforms, transform_name)

    def classify(self, image: numpy.ndarray) -> str:
        with torch.no_grad():
            input = self.transform(image)  # [CHW]
            input = input.reshape(1, *input.shape)  # [NCHW]
            self.model.eval()
            output: torch.Tensor = self.model(input)
            idx = self.output_to_idx(output)
            idx = idx.reshape(1)
            idx = idx.item()
        return self.classes[idx]

    def state_dict(self, **kwargs):
        d = {}
        d['model_name'] = self.model_name
        d['classes'] = self.classes
        d['transform_name'] = self.transform_name
        d['model_state_dict'] = self.model.state_dict(**kwargs)
        d['name'] = self.__class__.__name__
        return d

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        model_name = state_dict['model_name']
        classes = state_dict['classes']
        transform_name = state_dict['transform_name']
        self.model_name = model_name
        self.classes = classes
        self.transform_name = transform_name
        if model_name is not None:
            self.model: torch.nn.Module = models.get_model(model_name, len(classes), pretrained=False)
        if transform_name is not None:
            self.transform: typing.Callable[[typing.Union[PIL.Image.Image, numpy.ndarray]], torch.Tensor] = getattr(transforms, transform_name)
        self.model.load_state_dict(state_dict['model_state_dict'], strict=strict)


class ClassifierArgmax(Classifier):
    def __init__(
        self,
        model_name: str = None,
        classes: typing.List[str] = None,
        transform_name: str = None,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(model_name, classes, transform_name, functools.partial(torch.argmax, dim=-1), pretrained, **kwargs)


class EnsembleClassifier:
    def __init__(self, classifiers: typing.List[Classifier], method=None, threshold=0, prelim: Classifier = None, training_data_dic=None) -> None:
        assert len(classifiers) > 0
        classes = classifiers[0].classes
        for classifier in classifiers:
            assert classifier.classes == classes
        assert method in (None, 'ThreadPoolExecutor')

        self.classifiers = classifiers
        self.classes = classes
        self.softmax = torch.nn.Softmax(dim=-1)
        self.output_to_idx = functools.partial(torch.max, dim=-1)
        self.method = method
        self.threshold = threshold
        self.prelim = prelim
        self.training_data_dic = [] if training_data_dic is None else training_data_dic

    def fn(self, classifier: Classifier, image: numpy.ndarray):
        input = classifier.transform(image)  # [CHW]
        input = input.reshape(1, *input.shape)  # [NCHW]
        classifier.model.eval()
        output: torch.Tensor = classifier.model(input)
        return self.softmax(output)

    def classify(self, image: numpy.ndarray) -> str:
        if self.prelim is not None:
            c = self.prelim.classify(image)
            if c != 'isnull' and c not in self.training_data_dic:
                print('prelim', c)
                return c
        with torch.no_grad():
            if self.method == 'ThreadPoolExecutor':
                prob = 0
                futures: typing.List[concurrent.futures.Future] = []
                with concurrent.futures.ThreadPoolExecutor(len(self.classifiers)) as executor:
                    for classifier in self.classifiers:
                        futures.append(executor.submit(self.fn, classifier, image))
                for future in futures:
                    prob += future.result()
                prob = prob / len(self.classifiers)
            elif self.method is None:
                prob = 0
                for classifier in self.classifiers:
                    prob += self.fn(classifier, image)
                prob = prob / len(self.classifiers)
            max_prob, idx = self.output_to_idx(prob)
            if max_prob < self.threshold:
                c = 'isnull'
            else:
                idx = idx.reshape(1)
                idx = idx.item()
                c = self.classes[idx]
        return c
