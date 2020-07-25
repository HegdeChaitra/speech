import torch

def net_evaluation(net, kf, x, y, n=10):
    top1_correct = 0
    topn_correct = 0
    total = 0

    net.eval()
    for _, index in kf:
        y_batch = torch.autograd.Variable(torch.from_numpy(y[index]).long())
        x_batch = torch.autograd.Variable(torch.from_numpy(x[index]).float())
        
        if net.device == 'gpu':
            y_batch = y_batch.cuda()

        output = net(x_batch)
        total += y_batch.size(0)

        _, top1_predicted = torch.max(output, dim=1)
        top1_correct += int((top1_predicted == y_batch).sum()) 
    
        _, topn_predicted = torch.topk(output, k=n, dim=1, largest=True)
        for col in range(n):
            topn_correct += int((topn_predicted[:,col]==y_batch).sum())

    #print("top1_correct:", int(top1_correct))
    #print("topn_correct:", int(topn_correct))
    #print("total:", int(total))
    net.train()
    top1_acc = top1_correct/total
    topn_acc = topn_correct/total

    print("top1_acc", 100*top1_acc)
    print("topn_acc", 100*topn_acc)

    return top1_acc, topn_acc 



