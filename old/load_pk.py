import pickle
import argparse

parser = argparse.ArgumentParser(description='Pytorch "temporal name"')
parser.add_argument('-p', type=str, help='the path of weight')

def main():
    args = parser.parse_args()
    path = args.p
    with open(path, 'rb') as f:
        log_dict = pickle.load(f)

    print('train_acc1')
    print(log_dict['train_acc1'])

    print('val_acc1')
    print(log_dict['val_acc1'])

main()