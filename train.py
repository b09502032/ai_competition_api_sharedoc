import itertools
import json
import pathlib
import shutil
import sys

import sklearn.metrics
import torch
import torch.utils.data
import torchvision
import tqdm
import RandAugment.smooth_ce

import classifiers
import transforms
import util

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', default=64, type=int, help='how many samples per batch to load')
    parser.add_argument('--num-workers', default=8, type=int, help='how many subprocesses to use for data loading')
    parser.add_argument('--drop-last', action='store_true', help='drop the last incomplete train batch')
    parser.add_argument('--seed', default=7122, type=int, help=' ')
    parser.add_argument('--device')
    parser.add_argument('--train-dataset', nargs='+', required=True)
    parser.add_argument('--train-transform-name', nargs='+', metavar='NAME')
    parser.add_argument('--validation-dataset', nargs='+')
    parser.add_argument('--validation-transform-name', required=True, metavar='NAME')
    parser.add_argument('--model-name', required=True, metavar='NAME')
    parser.add_argument('--aux-logits-weight', default=0.4, type=float, help='aux logits weight for inception_v3')
    parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing factor')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--num-epochs', default=64, type=int, help=' ')
    parser.add_argument('--log-directory', help='directory to save models and logs', metavar='DIRECTORY')
    args = parser.parse_args()

    batch_size = args.batch_size
    num_workers = args.num_workers
    drop_last = args.drop_last
    seed = args.seed
    util.same_seeds(seed)
    if args.device is None:
        device = util.get_device()
    else:
        device = torch.device(args.device)

    train_dataset = args.train_dataset
    if args.train_transform_name is None:
        train_transform = itertools.cycle([None])
    else:
        train_transform = itertools.cycle([getattr(transforms, transform_name) for transform_name in args.train_transform_name])
    train_dataset = torch.utils.data.ConcatDataset([torchvision.datasets.ImageFolder(root, next(train_transform)) for root in train_dataset])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True, num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    validation_dataset = args.validation_dataset
    validation_transform_name = args.validation_transform_name
    validation_loader = None
    if validation_dataset is not None:
        validation_transform = None
        if validation_transform_name is not None:
            validation_transform = getattr(transforms, validation_transform_name)
        validation_dataset = torch.utils.data.ConcatDataset([torchvision.datasets.ImageFolder(root, validation_transform) for root in validation_dataset])
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size, False, num_workers=num_workers, pin_memory=True)

    class_to_idx = train_dataset.datasets[0].class_to_idx
    for dataset in train_dataset.datasets:
        assert dataset.class_to_idx == class_to_idx
    if validation_dataset is not None:
        for dataset in validation_dataset.datasets:
            assert dataset.class_to_idx == class_to_idx

    model_name = args.model_name
    classes = train_dataset.datasets[0].classes
    classifier = classifiers.ClassifierArgmax(model_name, classes, validation_transform_name, True)
    classifier.model.to(device)
    label_smoothing = args.label_smoothing
    lr = args.lr
    criterion = RandAugment.smooth_ce.SmoothCrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(classifier.model.parameters(), lr=lr)

    aux_logits_weight = args.aux_logits_weight

    log_directory = args.log_directory
    model_directory = None
    files = (sys.stdout, )
    if log_directory is not None:
        log_directory = pathlib.Path(log_directory)
        if log_directory.is_dir():
            a = input('Recursively delete {}? [Y/n] '.format(repr(str(log_directory))))
            if a in 'Yy':
                shutil.rmtree(log_directory)
        elif log_directory.is_file():
            a = input('Remove {}? [Y/n] '.format(repr(str(log_directory))))
            if a in 'Yy':
                log_directory.unlink()
        log_directory.mkdir()
        model_directory = log_directory / 'models'
        model_directory.mkdir()
        with (log_directory / 'info.json').open('w') as fp:
            json.dump(vars(args), fp)
        files += (log_directory / 'log.txt', )
    logger = util.Logger(*files)
    best_valid_acc = float('-inf')
    best_valid_f1 = float('-inf')

    num_epochs = args.num_epochs
    width = len('{:d}'.format(num_epochs))
    for epoch in tqdm.tqdm(range(num_epochs)):
        classifier.model.train()
        train_loss = []
        train_accs = []
        preds = []
        gts = []
        for batch in train_loader:
            imgs, labels = batch
            imgs: torch.Tensor
            labels: torch.Tensor

            output = classifier.model(imgs.to(device))
            if isinstance(output, tuple):
                if model_name == 'inception_v3':
                    logits, aux_logits = output
                    loss1: torch.Tensor = criterion(logits, labels.to(device))
                    loss2: torch.Tensor = criterion(aux_logits, labels.to(device))
                    loss = loss1 + aux_logits_weight * loss2
                else:
                    raise NotImplementedError
            elif isinstance(output, torch.Tensor):
                logits = output
                loss: torch.Tensor = criterion(logits, labels.to(device))
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=-1)
            acc = pred == labels.to(device)

            train_loss.append(loss.item())
            train_accs += acc.tolist()
            preds += pred.tolist()
            gts += labels.tolist()
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        macro_f1 = sklearn.metrics.f1_score(y_true=gts, y_pred=preds, labels=list(set(gts)), average='macro')
        logger.print(
            '[ Train | {:0{width}d}/{:0{width}d} ] loss = {:.5f}, acc = {:.5f}, f1 = {:.5f}'.format(
                epoch + 1, num_epochs, train_loss, train_acc, macro_f1, width=width
            ),
            end=''
        )

        if validation_loader is not None:
            classifier.model.eval()
            valid_loss = []
            valid_accs = []
            preds = []
            gts = []
            confidences = []
            gt_confidences = []
            for batch in validation_loader:
                imgs, labels = batch
                imgs: torch.Tensor
                labels: torch.Tensor

                with torch.no_grad():
                    logits: torch.Tensor = classifier.model(imgs.to(device))
                loss = criterion(logits, labels.to(device))

                pred = logits.argmax(dim=-1)
                acc = pred == labels.to(device)
                softmax = logits.softmax(-1)
                confidence = softmax[torch.arange(pred.shape[0]), pred]
                gt_confidence = softmax[torch.arange(labels.shape[0]), labels]

                valid_loss.append(loss.item())
                valid_accs += acc.tolist()
                preds += pred.tolist()
                gts += labels.tolist()
                confidences += confidence.tolist()
                gt_confidences += gt_confidence.tolist()
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)
            macro_f1 = sklearn.metrics.f1_score(y_true=gts, y_pred=preds, labels=list(set(gts)), average='macro')
            is_best_valid_acc = ' '
            if valid_acc > best_valid_acc:
                is_best_valid_acc = '*'
                best_valid_acc = valid_acc
                if model_directory is not None:
                    torch.save(classifier.state_dict(), model_directory / 'best_accuracy.ckpt')
            is_best_valid_f1 = ' '
            if macro_f1 > best_valid_f1:
                is_best_valid_f1 = '*'
                best_valid_f1 = macro_f1
                if model_directory is not None:
                    torch.save(classifier.state_dict(), model_directory / 'best_macro_average_f1.ckpt')
            logger.print(
                ' [ Valid | {:0{width}d}/{:0{width}d} ] loss = {:.5f}, acc ={}{:.5f}, f1 ={}{:.5f}'.format(
                    epoch + 1, num_epochs, valid_loss, is_best_valid_acc, valid_acc, is_best_valid_f1, macro_f1, width=width
                ),
                end=''
            )

        if model_directory is not None:
            torch.save(classifier.state_dict(), model_directory / 'last.ckpt')
        logger.print()
