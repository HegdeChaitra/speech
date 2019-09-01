import sys
import csv
import torch
from torch.autograd import Variable
import time


def net_evaluation(net, kf, device):
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = 0
#     net = torch.load(net)
    net.eval()
    s = time.time()
    for data in kf:
        x_batch = Variable(data[0]).to(device)
        l = Variable(data[1]).to(device)
        y_batch = Variable(data[2]).long().to(device)

        output,y_sm_out,w_attn,_ = net(x_batch)
        total += y_batch.size(0)

        _, top1_predicted = torch.max(output, dim=1)
        top1_correct += int((top1_predicted == y_batch).sum()) 
    
        _, topn_predicted = torch.topk(output, k=5, dim=1, largest=True)
        for col in range(5):
            top5_correct += int((topn_predicted[:,col]==y_batch).sum())
            
        _, topn_predicted = torch.topk(output, k=10, dim=1, largest=True)
        for col in range(10):
            top10_correct += int((topn_predicted[:,col]==y_batch).sum())
            
    print("time taken=",time.time()-s)
    top1_acc = top1_correct/total
    top5_acc = top5_correct/total
    top10_acc = top10_correct/total
    print("top1_acc", 100*top1_acc)
    print("top5_acc", 100*top5_acc)
    print("top10_acc",100*top10_acc)

def train_model(optimizer, loss_fun, model, loaders, device, save_name,save_path, num_epochs=60):
    best_score = 0
    best_au = 0
    loss_hist = {'train': [], 'validate': []}
    acc_hist = {'train': [], 'validate': []}
    for epoch in range(num_epochs):
        for ex, phase in enumerate(['train', 'validate']):
            start = time.time()
            total = 0
            top1_correct = 0
            running_loss = 0
            running_total = 0
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
            for data in loaders[ex]:
                optimizer.zero_grad()

                x = Variable(data[0]).to(device)
                l = Variable(data[1])
                y = Variable(data[2]).long().to(device)
#                 print(x.size(),l.size(),y.size())
                y_out,y_sm_out,at_w,att_out = model(x)
                total += y.size(0)
                _, top1_predicted = torch.max(y_out, dim=1)
                top1_correct += int((top1_predicted == y).sum())
                loss = loss_fun(y_out, y)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                N = y.size(0)
                running_loss += loss.item() * N
                running_total += N
            epoch_loss = running_loss / running_total
            epoch_acc = top1_correct / total
            loss_hist[phase].append(epoch_loss)
            acc_hist[phase].append(epoch_acc)
            print("epoch {} {} loss = {}, accurancy = {} time = {}".format(epoch, phase, epoch_loss, epoch_acc,
                                                                           time.time() - start))
        if phase == 'validate' and epoch_acc >= best_score:
            best_score = epoch_acc
            torch.save(model, save_name)
    print("Training completed. Best accuracy is {}".format(best_score))
    return loss_hist, acc_hist
