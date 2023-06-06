from prototypical_loss import prototypical_loss as loss_fn

import numpy as np
import torch
import os

def train(args, tr_dataloader, model, optim, lr_scheduler, device, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join('./few_shot', 'best_model.pth')
    last_model_path = os.path.join('./few_shot', 'last_model.pth')

    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for idx, batch in enumerate(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            # print(model_output.shape)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=args.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-args.iterations:])
        avg_acc = np.mean(train_acc[-args.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=args.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-args.iterations:])
        avg_acc = np.mean(val_acc[-args.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    # for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
    #     save_list_to_file(os.path.join(args.experiment_root,
    #                                    name + '.txt'), locals()[name])

    # return best_acc, train_loss, train_acc, val_loss, val_acc
    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(args, test_dataloader, model, device):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=args.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


# def eval(opt):
#     '''
#     Initialize everything and train
#     '''
#     options = get_parser().parse_args()

#     if torch.cuda.is_available() and not options.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#     init_seed(options)
#     test_dataloader = init_dataset(options)[-1]
#     model = init_protonet(options)
#     model_path = os.path.join(opt.experiment_root, 'best_model.pth')
#     model.load_state_dict(torch.load(model_path))

#     test(opt=options,
#          test_dataloader=test_dataloader,
#          model=model)

def few_shot_train(tr_dataloader, val_dataloader, test_dataloader, model, optim, lr_scheduler, args, device):
    '''
    Initialize everything and train
    '''
    res = train(args=args,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler,
                device=device)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    # best_acc, train_loss, train_acc, val_loss, val_acc = res
    # print(f'best_acc:{best_acc}, train_acc:{train_acc}, val_acc:{val_acc}')

    print('Testing with last model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model, device=device)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model, device=device)