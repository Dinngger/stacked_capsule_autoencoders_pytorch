import os
import pdb
import shutil
import sys

import traceback
from absl import flags
from absl import logging

import torch
# import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from models.autoencoder import ImageAutoencoder
from data import getMnist


flags.DEFINE_string('logdir', 'checkpoints/',
                    'Log and checkpoint directory for the experiment.')
flags.DEFINE_integer('report_loss_steps', 50, '')
flags.DEFINE_boolean('overwrite', True, 'Overwrites any existing run of the '
                     'same name if True; otherwise it tries to restore the '
                     'model if a checkpoint exists.')

flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('test_batch_size', 512, 'batch size')
flags.DEFINE_integer('epochs', 1000, 'number of epochs to train')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('gamma', 1, 'Learning rate schedule rate.')


flags.DEFINE_integer('template_size', 11, 'Template size.')
flags.DEFINE_integer('n_part_caps', 16, 'Number of part capsules.')
flags.DEFINE_integer('d_part_pose', 6, 'Part caps\' dimensionality.')
flags.DEFINE_integer('d_part_features', 16, 'Number of special '
                     'features.')

flags.DEFINE_integer('n_channels', 1, 'Number of input channels.')

flags.DEFINE_integer('n_obj_caps', 10, 'Number of object capsules.')
flags.DEFINE_integer('n_obj_caps_params', 32, 'Dimensionality of object caps '
                     'feature vector.')

flags.DEFINE_boolean('colorize_templates', False, 'Whether to infer template '
                     'color from input.')
flags.DEFINE_boolean('use_alpha_channel', False, 'Learns per-pixel mixing '
                     'proportions for every template; otherwise mixing '
                     'probabilities are constrained to have the same value as '
                     'image pixels.')

flags.DEFINE_string('template_nonlin', 'relu1', 'Nonlinearity used to normalize'
                    ' part templates.')
flags.DEFINE_string('color_nonlin', 'relu1', 'Nonlinearity used to normalize'
                    ' template color (intensity) value.')

flags.DEFINE_float('prior_within_example_sparsity_weight', 1., 'Loss weight.')
flags.DEFINE_float('prior_between_example_sparsity_weight', 1., 'Loss weight.')
flags.DEFINE_float('posterior_within_example_sparsity_weight', 10.,
                   'Loss weight.')
flags.DEFINE_float('posterior_between_example_sparsity_weight', 10.,
                   'Loss weight.')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            res = model(data.to(device), target.to(device), device=device)
            # sum up batch loss
            test_loss += model.loss(res).item() * len(data)
            # get the index of the max log-probability
            correct += res.best_cls_acc.item() * len(data)

    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    return test_loss, correct


def train(args, model, device, train_loader, test_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        res = model(data.to(device), target.to(device), device=device)
        loss = model.loss(res)
        loss.backward()
        optimizer.step()

        if batch_idx % args.report_loss_steps == 0:
            test_loss, test_acc = test(model, device, test_loader)
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalars('Loss', {'train': loss.item(), 'test': test_loss}, global_step)
            writer.add_scalars('Accuracy', {'train': res.best_cls_acc.item(), 'test': test_acc}, global_step)
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}\ttest_loss: {:.6f}\tbest_cls_acc: {:.6f}\ttest_acc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), test_loss, res.best_cls_acc.item(), test_acc))


def main():
    config = flags.FLAGS
    config(sys.argv)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    logdir = config.logdir
    logging.info('logdir: %s', logdir)

    if os.path.exists(logdir) and config.overwrite:
        logging.info(
            '"overwrite" is set to True. Deleting logdir at "%s".', logdir)
        shutil.rmtree(logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    writer = SummaryWriter(logdir)

    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    train_loader, test_loader = getMnist(train_kwargs, test_kwargs)

    model = ImageAutoencoder(config=config).to(device)
    eps = 1e-2 / float(config.batch_size) ** 2
    optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=eps)

    scheduler = ExponentialLR(optimizer, gamma=config.gamma)

    for epoch in range(1, config.epochs + 1):
        logging.info("lr={}".format(scheduler.get_last_lr()[0]))
        train(config, model, device, train_loader, test_loader, optimizer, epoch, writer)
        scheduler.step()

    torch.save(model.state_dict(), logdir + "mnist_scae.pt")
    writer.close()


if __name__ == '__main__':
    try:
        logging.set_verbosity(logging.INFO)
        main()
    except Exception as err:
        config = flags.FLAGS
        last_traceback = sys.exc_info()[2]
        traceback.print_tb(last_traceback)
        print(err)
        pdb.post_mortem(last_traceback)
