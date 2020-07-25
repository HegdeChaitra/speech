import argparse
from args import parse_args
args = parse_args()

import os
import pickle
import time

import numpy as np
import torch.nn.parallel
from sklearn.model_selection import StratifiedShuffleSplit
from torch import optim

from model import MLP
import eval

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def train(lr=args.lr,
          n_hidden=args.n_hidden,
          batch_size=args.batch_size,
          dropout = args.dropout,
          valid_freq=3000,
          disp_freq=1000,
          save_freq=100000,
          max_epochs=args.n_epoch,
          patience=15,
          save_name=args.save_name,
          save_dir=args.save_dir,
          device=args.device):
    # Load train and valid dataset
    print('loading train')
    with open(args.train_path, 'rb') as f:
        train_val_y = pickle.load(f)
        train_val_x = pickle.load(f)

    print('loading english test')
    with open(args.en_test_path, 'rb') as f:
        en_test_y = pickle.load(f)
        en_test_x = pickle.load(f)

    print('loading french test')
    with open(args.fr_test_path, 'rb') as f:
        fr_test_y = pickle.load(f)
        fr_test_x = pickle.load(f)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1125)
    for train_index, test_index in sss.split(train_val_x, train_val_y):
        train_y = train_val_y[train_index]
        train_x = train_val_x[train_index]
        valid_y = train_val_y[test_index]
        valid_x = train_val_x[test_index]

    print('Number of training sample: %d' % train_x.shape[0])
    print('Number of validation sample: %d' % valid_x.shape[0])
    print('Number of english testing sample: %d' % en_test_x.shape[0])
    print('Number of french testing sample: %d' % fr_test_x.shape[0])
    print('-' * 100)

    kf_valid = get_minibatches_idx(len(valid_y), batch_size)
    kf_en_test = get_minibatches_idx(len(en_test_y), batch_size)
    kf_fr_test = get_minibatches_idx(len(fr_test_y), batch_size)

    # Loader parameter: use CUDA pinned memory for faster data loading
    pin_memory = (device == args.device)
    # Test set
    
    n_emb = train_x.shape[1]
    n_class = len(set(train_y))
    best_valid_acc = None
    bad_counter = 0

    uidx = 0  # the number of update done
    estop = False  # early stop switch
    net = MLP(n_mlp_layer=args.n_mlp_layers,
              n_hidden=args.n_hidden,
              dropout = args.dropout,
              n_class=n_class,
              n_emb=n_emb,
              device=args.device)

    if args.load_net != '':
        assert os.path.exists(args.load_net), 'Path to pretrained net does not exist'
        net.load_state_dict(torch.load(args.load_net))
        print('Load exists model stored at: ', args.load_net)

    if args.device == 'gpu':
        net = net.cuda()

    # Begin Training
    net.train()
    print('-' * 100)
    print('Model structure: ')
    print('MLP baseline')
    print(net.main)
    print('-' * 100)
    print('Parameters for tuning: ')
    print(net.state_dict().keys())
    print('-' * 100)

    # Define optimizer
    assert args.optimizer in ['SGD', 'Adam', "RMSprop", "LBFGS", "Rprop", "ASGD", "Adadelta", "Adagrad", "Adamax"], 'Please choose either SGD or Adam'
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(lr=lr, params=filter(lambda p: p.requires_grad, net.parameters()), momentum=0.9)
    else:
        optimizer = getattr(optim, args.optimizer)(params=filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    
    #lambda1 = lambda epoch: epoch // 30
    lambda2 = lambda epoch: 0.98 ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2])
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    try:
        for eidx in range(max_epochs):
            scheduler.step()
            # print('Training mode on: ' ,net.training)
            start_time = time.time()
            n_samples = 0
            # Get new shuffled index for the training set
            kf = get_minibatches_idx(len(train_y), batch_size, shuffle=True)

            for _, train_index in kf:
                # Remove gradient from previous batch
                #net.zero_grad()
                optimizer.zero_grad()
                uidx += 1
                y_batch = torch.autograd.Variable(torch.from_numpy(train_y[train_index]).long())
                x_batch = torch.autograd.Variable(torch.from_numpy(train_x[train_index]).float())
                if net.device == 'gpu':
                    y_batch = y_batch.cuda()
                scores = net.forward(x_batch)
                loss = net.loss(scores, y_batch)

                loss.backward()
                optimizer.step()
                n_samples += len(x_batch)
                gradient = 0

                # For logging gradient information
                for name, w in net.named_parameters():
                    if w.grad is not None:
                        w_grad = torch.norm(w.grad.data, 2) ** 2
                        gradient += w_grad
                gradient = gradient ** 0.5
                if np.mod(uidx, disp_freq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', loss.data[0], 'Gradient ', gradient)

                if save_name and np.mod(uidx, save_freq) == 0:
                    print('Saving...')
                    torch.save(net.state_dict(), '%s/%s_epoch%d_update%d.net' % (save_dir, save_name, eidx, uidx))

                if np.mod(uidx, valid_freq) == 0:
                    print("="*50)
                    print('Evaluation on validation set: ')
                    kf_valid = get_minibatches_idx(len(valid_y), batch_size)
                    top_1_acc, top_n_acc = eval.net_evaluation(net, kf_valid, valid_x, valid_y)
                    #scheduler.step(top_1_acc)
 
                    # Save best performance state_dict for testing
                    if best_valid_acc is None:
                        best_valid_acc = top_1_acc
                        best_state_dict = net.state_dict()
                        torch.save(best_state_dict, '%s/%s_best.net' % (save_dir, save_name))
                    else:
                        if top_1_acc > best_valid_acc:
                            print('Best validation performance so far, saving model parameters')
                            print("*" * 50)
                            bad_counter = 0  # reset counter
                            best_valid_acc = top_1_acc
                            best_state_dict = net.state_dict()
                            torch.save(best_state_dict, '%s/%s_best.net' % (save_dir, save_name))
                        else:
                            bad_counter += 1
                            print('Validation accuracy: ', 100*top_1_acc)
                            print('Getting worse, patience left: ', patience - bad_counter)
                            print('Best validation accuracy  now: ', 100*best_valid_acc)
                            # Learning rate annealing
                            lr /= args.lr_anneal
                            print('Learning rate annealed to: ', lr)
                            print('*' * 100)
                            if args.optimizer == 'SGD':
                                optimizer = optim.SGD(lr=lr, params=filter(lambda p: p.requires_grad, net.parameters()), momentum=0.9)
                            else:
                                optimizer = getattr(optim,args.optimizer)(params = filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
                            if bad_counter > patience:
                                print('-' * 100)
                                print('Early Stop!')
                                estop = True
                                break

            epoch_time = time.time() - start_time
            print('Epoch processing time: %.2f s' % epoch_time)
            print('Seen %d samples' % n_samples)
            if estop:
                break
        print('-' * 100)
        print('Training finish')
        best_state_dict = torch.load('%s/%s_best.net' % (save_dir, save_name))
        torch.save(net.state_dict(), '%s/%s_final.net' % (save_dir, save_name))
        net.load_state_dict(best_state_dict)

        # add self connection
        print('Evaluation on validation set: ')
        kf_valid = get_minibatches_idx(len(valid_y), batch_size)
        eval.net_evaluation(net, kf_valid, valid_x, valid_y)

        # Evaluate model on test set
        print('Evaluation on test set: ')
        print('Evaluation on English testset: ')
        eval.net_evaluation(net, kf_en_test, en_test_x, en_test_y)
        print('Evaluation on French testset: ')
        eval.net_evaluation(net, kf_fr_test, fr_test_x, fr_test_y)
    except KeyboardInterrupt:
        print('-' * 100)
        print("Training interrupted, saving final model...")
        best_state_dict = torch.load('%s/%s_best.net' % (save_dir, save_name))
        torch.save(net.state_dict(), '%s/%s_final.net' % (save_dir, save_name))
        net.load_state_dict(best_state_dict)
        print('Evaluation on validation set: ')
        kf_valid = get_minibatches_idx(len(valid_y), batch_size)
        eval.net_evaluation(net, kf_valid, valid_x, valid_y)

        # Evaluate model on test set
        print('Evaluation on English testset: ')
        eval.net_evaluation(net, kf_en_test, en_test_x, en_test_y)
        print('Evaluation on French testset: ')
        eval.net_evaluation(net, kf_fr_test, fr_test_x, fr_test_y)


if __name__ == '__main__':
    
    try:
        os.makedirs(args.save_dir)
    except OSError:
        pass

    print(args.save_name)
    print('Arguments:   ')
    print(args)
    print('-' * 100)


    train()
