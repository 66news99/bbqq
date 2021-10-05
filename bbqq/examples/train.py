import torch
from tqdm.autonotebook import tqdm


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


def train(EPOCHS=None, model=None, train_dataloader=None, test_dataloader=None, optimizer=None, log_interval=None):
    for e_idx, epoch in enumerate(range(EPOCHS)):
        losses = list()
        train_acc = 0.0
        test_acc = 0.0

        model.train()
        for b_idx, batch in enumerate(tqdm(train_dataloader)):
            X, y = batch
            loss = model.training_step(X, y)
            y_hat = model.forward(X)
            optimizer.zero_grad()  # resetting the gradients.
            loss.backward(retain_graph=True)  # backprop the loss
            optimizer.step()  # gradient step
            train_acc += calc_accuracy(y_hat, y)
            losses.append(loss.item())
        avg_loss = (sum(losses) / len(losses))
        if b_idx % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e_idx, b_idx + 1, loss.data.cpu().numpy(), train_acc / (b_idx + 1)))
    print("epoch {} train acc {}".format(EPOCHS, train_acc / (b_idx + 1)))

    model.eval()
    for b_idx, batch in enumerate(tqdm(test_dataloader)):
        X, y = batch
        y_hat = model.forward(X)
        test_acc += calc_accuracy(y_hat, y)
    print(f"epoch {e_idx+1} test acc {test_acc}" )