import torch
from tqdm.autonotebook import tqdm

from bbqq.f1_score import f1_loss


def train_test(EPOCHS=None, model=None, train_dataloader=None, test_dataloader=None, optimizer=None, log_interval=None,
          max_grad_norm=None, scheduler=None):
    for e_idx, epoch in enumerate(range(EPOCHS)):
        losses = list()
        train_f1 = 0.0
        test_f1 = 0.0

        model.train()
        for b_idx, batch in enumerate(tqdm(train_dataloader)):
            X, y = batch
            loss = model.training_step(X, y)
            y_hat = model.predict(X)
            optimizer.zero_grad()  # resetting the gradients.
            loss.backward(retain_graph=True)  # backprop the loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()  # gradient step
            scheduler.step()
            train_f1 += f1_loss(y, y_hat)
            losses.append(loss.item())
        avg_loss = (sum(losses) / len(losses))
        if b_idx % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e_idx, b_idx + 1, loss.data.cpu().numpy(), train_f1 / (b_idx + 1)))
    print("epoch {} train acc {}".format(EPOCHS, train_f1 / (b_idx + 1)))

    model.eval()
    for b_idx, batch in enumerate(tqdm(test_dataloader)):
        X, y = batch
        y_hat = model.forward(X)
        test_f1 += f1_loss(y, y_hat)
    print(f"epoch {e_idx+1} test acc {test_f1}" )